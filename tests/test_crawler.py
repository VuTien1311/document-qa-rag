"""Test crawler với một website giả (httpx.MockTransport): không gửi request thật nào ra ngoài."""

import httpx
import pytest

from rag.crawler import Fetcher, crawl, sitemap_urls
from rag.loaders import load_document
from rag.pipeline import source_label

SITE = "https://site.test"
SITEMAP_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
SITEMAP_NS_HTTPS = 'xmlns="https://www.sitemaps.org/schemas/sitemap/0.9"'  # không chuẩn nhưng có site dùng
PAGE_TEXT = "Học phí mỗi học kỳ của chuyên ngành Trí tuệ nhân tạo là 31.600.000 đồng. " * 6

ROUTES = {
    "/robots.txt": "User-agent: *\nDisallow: /private\n",
    "/sitemap.xml": f"<sitemapindex {SITEMAP_NS}><sitemap><loc>{SITE}/pages.xml</loc></sitemap></sitemapindex>",
    "/pages.xml": f"<urlset {SITEMAP_NS_HTTPS}>"
    + "".join(f"<url><loc>{SITE}{p}</loc></url>" for p in ["/hoc-phi/", "/private/bang-diem/", "/thank-you/"])
    + "</urlset>",
    "/hoc-phi/": "<html><head><title>Học phí</title></head><body>"
    "<nav><a href='/'>Trang chủ</a> | <a href='/lien-he'>Liên hệ</a></nav>"
    f"<article><h1>Học phí</h1><p>{PAGE_TEXT}</p></article>"
    "<footer>© Trường mẫu</footer></body></html>",
}


@pytest.fixture
def fetcher():
    def handler(request: httpx.Request) -> httpx.Response:
        body = ROUTES.get(request.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    return Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_sitemap_index_is_followed(fetcher):
    assert sitemap_urls(fetcher, f"{SITE}/sitemap.xml") == [
        f"{SITE}/hoc-phi/",
        f"{SITE}/private/bang-diem/",
        f"{SITE}/thank-you/",
    ]


def test_robots_txt_is_respected(fetcher):
    assert fetcher.allowed(f"{SITE}/hoc-phi/")
    assert not fetcher.allowed(f"{SITE}/private/bang-diem/")
    with pytest.raises(PermissionError):
        fetcher.get(f"{SITE}/private/bang-diem/")


def test_crawl_saves_main_content_with_source_url(fetcher, tmp_path):
    results = []
    saved = crawl(
        tmp_path,
        sitemaps=[f"{SITE}/sitemap.xml"],
        exclude="thank-you",
        fetcher=fetcher,
        on_page=lambda url, path, status: results.append((url, status)),
    )
    assert [url for url, _ in results] == [f"{SITE}/hoc-phi/", f"{SITE}/private/bang-diem/"]
    assert "robots.txt" in results[1][1]
    assert len(saved) == 1

    doc = load_document(saved[0])
    [section] = doc.sections
    assert doc.title == "Học phí"
    assert section.location == f"{SITE}/hoc-phi/"
    assert "31.600.000 đồng" in section.text
    assert "Liên hệ" not in section.text  # menu bị loại
    assert source_label({"filename": saved[0].name, "location": section.location}) == f"{SITE}/hoc-phi/"


def test_markdown_without_front_matter_is_plain_text(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("---\nkhông phải front matter vì không đóng", encoding="utf-8")
    [section] = load_document(path).sections
    assert section.location == "" and "không phải front matter" in section.text
