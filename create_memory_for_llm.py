# create_memory_for_llm.py
import os
import glob
import pickle
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

os.makedirs(DB_PATH, exist_ok=True)


def read_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def gather_documents(data_dir):
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    docs = []
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        pages = read_pdf_pages(pdf_path)
        for page in pages:
            page_chunks = chunk_text(page["text"])
            for chunk in page_chunks:
                docs.append({
                    "page_content": chunk,
                    "metadata": {"source": filename, "page": page["page"]}
                })
    return docs


def main():
    print(f"[INFO] Reading PDFs from: {DATA_PATH}")
    docs = gather_documents(DATA_PATH)
    print(f"[INFO] Collected {len(docs)} chunks.")
    if not docs:
        print("[WARN] No docs found. Put PDFs in the data/ directory.")
        return
    out_file = os.path.join(DB_PATH, "index.pkl")
    with open(out_file, "wb") as f:
        pickle.dump({"docs": docs}, f)
    print(f"[SUCCESS] Saved metadata to {out_file}")


if __name__ == "__main__":
    main()


