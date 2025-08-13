# create_memory_for_llm.py
import os
import glob
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers model id
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 64))

os.makedirs(DB_PATH, exist_ok=True)


def read_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        # normalize whitespace
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
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
            page_text = page["text"]
            # split page text into chunks
            page_chunks = chunk_text(page_text)
            for chunk in page_chunks:
                docs.append({
                    "page_content": chunk,
                    "metadata": {"source": filename, "page": page["page"]}
                })
    return docs


def build_faiss_index(docs, model_name=EMBED_MODEL, db_path=DB_PATH):
    print(f"[INFO] Loading embedding model: sentence-transformers/{model_name}")
    embedder = SentenceTransformer(model_name)

    texts = [d["page_content"] for d in docs]
    metas = [d["metadata"] for d in docs]

    dim = embedder.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}")

    # compute embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        emb = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        print(f"[INFO] Embedded {i + len(batch_texts)}/{len(texts)} chunks")
    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # normalize for cosine similarity (inner product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # create FAISS index
    index = faiss.IndexFlatIP(dim)  # inner product with normalized vectors = cosine
    index.add(embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors")

    # save index and docs metadata
    faiss.write_index(index, os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "index.pkl"), "wb") as f:
        pickle.dump({"docs": docs}, f)

    print(f"[SUCCESS] Saved index to {db_path}")


def main():
    docs = gather_documents(DATA_PATH)
    print(f"[INFO] Collected {len(docs)} document chunks from PDFs in {DATA_PATH}")
    if not docs:
        print("[WARN] No documents found. Put PDFs in the 'data/' folder and re-run.")
        return
    build_faiss_index(docs)


if __name__ == "__main__":
    main()
