# Document QA RAG

Trợ lý hỏi đáp tài liệu dùng **RAG (Retrieval-Augmented Generation)**: tải lên PDF, DOCX, TXT rồi đặt câu hỏi,
câu trả lời luôn kèm **trích dẫn nguồn** theo tên file và trang/mục để người dùng tự kiểm chứng.

- Hỗ trợ PDF (trích dẫn theo trang), DOCX (theo mục Heading, đọc được cả bảng), TXT/Markdown
- Tìm kiếm ngữ nghĩa với vector embedding + ChromaDB, có thể giới hạn phạm vi hỏi trong một số tài liệu
- Câu trả lời stream từng chữ, mỗi ý gắn số nguồn `[1]`, `[2]`, bấm vào để xem đoạn gốc
- Chạy miễn phí: **Gemini API** (gói free) hoặc **Ollama** (model chạy trên máy), đổi bằng một biến cấu hình

## Kiến trúc

```mermaid
flowchart LR
    subgraph Ingest["1. Nạp tài liệu"]
        A[PDF / DOCX / TXT] --> B[Đọc + chuẩn hoá<br/>loaders.py]
        B --> C[Chia chunk<br/>chunker.py]
        C --> D[Embedding]
        D --> E[(ChromaDB)]
    end
    subgraph QA["2. Hỏi đáp"]
        Q[Câu hỏi] --> QE[Embedding]
        QE --> R[Top-K chunk gần nhất]
        R --> L[LLM]
        L --> ANS[Câu trả lời + nguồn trích dẫn]
    end
    E -.-> R
```

| Thành phần | Công nghệ |
|---|---|
| LLM | Gemini `gemini-3.5-flash-lite` (mặc định) hoặc Ollama `qwen2.5:7b` |
| Embedding | `gemini-embedding-2` (768 chiều) hoặc Ollama `bge-m3` |
| Vector DB | ChromaDB, khoảng cách cosine |
| Đọc tài liệu | PyMuPDF, python-docx |
| Giao diện | Streamlit |

## Cài đặt

Cần Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Tạo API key Gemini miễn phí tại https://aistudio.google.com/apikey, copy `.env.example` thành `.env` rồi điền:

```
GEMINI_API_KEY=your-key
```

> Gói free của Gemini giới hạn số request mỗi phút/ngày và dữ liệu gửi lên có thể được Google dùng để cải thiện
> sản phẩm. Với tài liệu nhạy cảm, hãy dùng Ollama (xem bên dưới).

## Chạy

```bash
streamlit run app.py
```

Mở http://localhost:8501, tải lên các file trong `sample_docs/` (tài liệu mẫu của một công ty hư cấu) và thử hỏi:

- Chính sách hoàn tiền của công ty như thế nào?
- Máy báo lỗi E2 thì phải làm gì?
- Bộ lọc HEPA được bảo hành bao lâu?
- Công ty có bán ô tô không? *(không có trong tài liệu, hệ thống phải trả lời là không tìm thấy)*

Hoặc dùng trên terminal:

```bash
python cli.py add sample_docs/Chinh_sach_cong_ty.docx sample_docs/Huong_dan_su_dung.pdf
python cli.py ask "Máy báo lỗi E2 thì phải làm gì?" --show-context
```

### Chạy local bằng Ollama (không cần API key)

Cài [Ollama](https://ollama.com/download), tải model rồi đặt `PROVIDER=ollama` trong `.env`:

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b
```

`qwen2.5:7b` cần khoảng 5 GB VRAM/RAM; máy yếu hơn dùng `LLM_MODEL=qwen2.5:3b`.

## Test

Test dùng embedding và LLM giả nên chạy được không cần API key:

```bash
pip install -r requirements-dev.txt
pytest
```

## Cấu trúc

```
app.py                  Giao diện Streamlit
cli.py                  Giao diện dòng lệnh
rag/
  config.py             Cấu hình (ghi đè bằng biến môi trường / .env)
  loaders.py            Đọc PDF theo trang, DOCX theo mục, TXT; chuẩn hoá Unicode
  chunker.py            Chia chunk đệ quy: đoạn > dòng > câu > từ, có chồng lấn
  llm.py                Chọn provider; phần còn lại không phụ thuộc nhà cung cấp model
  providers/gemini.py   Gemini API (tự retry khi bị giới hạn tần suất)
  providers/ollama.py   Ollama local
  store.py              ChromaDB + danh sách tài liệu
  pipeline.py           ingest() → retrieve() → generate() → cited_sources()
tests/                  pytest
sample_docs/            Tài liệu mẫu
```

## Cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PROVIDER` | `gemini` | `gemini` hoặc `ollama` |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Model sinh câu trả lời (`gemini-3.8-flash` trả lời tốt hơn, hạn mức free thấp hơn) |
| `EMBED_MODEL` | `gemini-embedding-2` | Model embedding |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Kích thước chunk và phần chồng lấn (ký tự) |
| `TOP_K` | `5` | Số chunk đưa cho LLM mỗi câu hỏi |

Mỗi model embedding có thư mục dữ liệu riêng trong `data/`, vì vector của các model khác nhau không so sánh được với
nhau. Đổi `CHUNK_SIZE` thì cần xoá `data/` và nạp lại tài liệu.

## Một số quyết định thiết kế

- **Chunk không vắt qua trang PDF**: mỗi trang được chia chunk riêng nên trích dẫn luôn đúng số trang.
- **DOCX trích dẫn theo mục**: file Word không lưu số trang, nên dùng tiêu đề (Heading) gần nhất làm vị trí.
- **Chuẩn hoá Unicode NFC**: cùng một chữ tiếng Việt có thể được lưu ở dạng dựng sẵn hoặc tổ hợp tuỳ phần mềm soạn
  thảo; không chuẩn hoá thì cùng một từ sẽ không khớp nhau khi tìm kiếm.
- **Chống bịa**: prompt yêu cầu chỉ dùng các đoạn trích, ghi nguồn sau mỗi ý và trả lời "không tìm thấy" khi thiếu
  thông tin; temperature thấp (0.1). Chỉ những nguồn được LLM thực sự trích dẫn mới hiện ra cho người dùng.
- **Bỏ qua file trùng**: tài liệu được định danh bằng hash nội dung nên upload lại cùng file không tạo dữ liệu trùng.

## Giới hạn và hướng phát triển

- [ ] Bộ câu hỏi đánh giá, đo Hit@K / MRR cho retrieval và độ đúng của câu trả lời
- [ ] Hybrid search (BM25 + vector) để bắt tốt từ khoá, mã số, tên riêng
- [ ] Reranker chấm lại các chunk trước khi đưa cho LLM
- [ ] Hiểu câu hỏi nối tiếp dựa trên lịch sử chat
- [ ] OCR cho PDF dạng ảnh scan, nạp tài liệu từ URL
- [ ] Tách backend FastAPI, đóng gói Docker, deploy
