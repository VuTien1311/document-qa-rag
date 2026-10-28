"""Thu thập trang web công khai thành file Markdown (kèm URL gốc) để nạp vào RAG.

Lịch sự với server: tuân thủ robots.txt và giãn cách giữa các request.
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

USER_AGENT = "document-qa-rag/0.1 (+https://github.com/VuTien1311/document-qa-rag)"


class Fetcher:
    """HTTP client tuân thủ robots.txt của từng site và chờ `delay` giây giữa 2 request."""

    def __init__(self, delay: float = 1.0, client: httpx.Client | None = None):
        self.delay = delay
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request = 0.0

    def get(self, url: str) -> httpx.Response:
        if not self.allowed(url):
            raise PermissionError(f"blocked by robots.txt: {url}")
        return self._get(url)

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in self._robots:
            response = self._get(f"{site}/robots.txt")
            rules = RobotFileParser()
            rules.parse(response.text.splitlines() if response.status_code == 200 else [])  # không có = cho phép hết
            self._robots[site] = rules
        return self._robots[site].can_fetch(USER_AGENT, url)

    def _get(self, url: str) -> httpx.Response:
        wait = self._last_request + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        return self.client.get(url)


def sitemap_urls(fetcher: Fetcher, sitemap_url: str) -> list[str]:
    """Danh sách URL trong sitemap; tự đi vào các sitemap con nếu đây là sitemap index."""
    root = ET.fromstring(fetcher.get(sitemap_url).content)
    locs = [el.text.strip() for el in root.iter() if _local_name(el.tag) == "loc" and el.text]
    if _local_name(root.tag) == "sitemapindex":
        return [url for child in locs for url in sitemap_urls(fetcher, child)]
    return locs


def _local_name(tag: str) -> str:
    """Bỏ phần namespace: chuẩn là http://www.sitemaps.org/..., nhưng có site ghi https://."""
    return tag.rsplit("}", 1)[-1]


def extract_page(html: str, url: str) -> tuple[str, str] | None:
    """Lấy (tiêu đề, nội dung chính dạng Markdown), bỏ menu/footer/quảng cáo. None nếu trang không có nội dung.

    Chạy cả chế độ mặc định lẫn favor_recall của trafilatura rồi giữ bản dài hơn: tuỳ cấu trúc trang,
    mỗi chế độ có thể bỏ sót những khối nội dung khác nhau.
    """
    candidates = [
        trafilatura.extract(html, url=url, output_format="markdown", include_tables=True, favor_recall=recall)
        for recall in (False, True)
    ]
    text = max(filter(None, candidates), key=len, default=None)
    if not text:
        return None
    meta = trafilatura.extract_metadata(html, default_url=url)
    return (meta.title if meta and meta.title else url), text


def save_page(out_dir: Path, url: str, title: str, text: str) -> Path:
    """Lưu thành Markdown, URL gốc ghi ở front matter để loader dùng làm nguồn trích dẫn."""
    parts = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{parts.netloc}{parts.path}".lower()).strip("-")
    path = out_dir / f"{slug}.md"
    path.write_text(f"---\ntitle: {title}\nurl: {url}\n---\n\n# {title}\n\n{text}\n", encoding="utf-8")
    return path


def crawl(
    out_dir: Path,
    sitemaps: list[str] = (),
    urls: list[str] = (),
    include: str | None = None,
    exclude: str | None = None,
    limit: int | None = None,
    fetcher: Fetcher | None = None,
    on_page: Callable[[str, Path | None, str], None] | None = None,
) -> list[Path]:
    """Tải các trang rồi lưu vào out_dir: URL chỉ định trong `urls` lấy hết, URL từ sitemap thì lọc bằng
    regex include/exclude."""
    fetcher = fetcher or Fetcher()
    from_sitemaps = [
        url
        for sitemap in sitemaps
        for url in sitemap_urls(fetcher, sitemap)
        if (not include or re.search(include, url)) and not (exclude and re.search(exclude, url))
    ]
    selected = list(dict.fromkeys([*urls, *from_sitemaps]))[:limit]  # bỏ trùng, giữ thứ tự

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for url in selected:
        try:
            response = fetcher.get(url)
            page = extract_page(response.text, url) if response.status_code == 200 else None
            path = save_page(out_dir, url, *page) if page else None
            status = "ok" if path else f"skipped (HTTP {response.status_code} or no main content)"
        except (httpx.HTTPError, PermissionError) as e:
            path, status = None, f"error: {e}"
        if path:
            saved.append(path)
        if on_page:
            on_page(url, path, status)
    return saved
