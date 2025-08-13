# create_memory_for_llm.py
"""
Creates FAISS vector index from PDFs under DATA_PATH and saves to DB_FAISS_PATH.

Produces:
  - vectorstore/db_faiss/index.faiss
  - vectorstore/db_faiss/index.pkl  (metadata + docs)
"""

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

# Config (can come from .env)
DATA_PATH = os.getenv("DATA_PATH", "data/")
DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers id
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 64))

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
    while start < len(text):
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


def build_faiss_index(docs, model_name=EMBED_MODEL, db_path=DB_PATH):
    print(f"[INFO] Loading embedding model: sentence-transformers/{model_name}")
    embedder = SentenceTransformer(model_name)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}")

    texts = [d["page_content"] for d in docs]
    metas = [d["metadata"] for d in docs]

    # embed in batches
    embeddings_list = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i:i + EMBED_BATCH_SIZE]
        emb = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(emb)
        print(f"[INFO] Embedded {i + len(batch_texts)}/{len(texts)}")
    if embeddings_list:
        embeddings = np.vstack(embeddings_list).astype("float32")
    else:
        embeddings = np.zeros((0, dim), dtype="float32")

    # normalize vectors for cosine similarity (inner product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # create FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] Built FAISS index with {index.ntotal} vectors")

    # save index and metadata
    faiss.write_index(index, os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "index.pkl"), "wb") as f:
        pickle.dump({"docs": docs}, f)

    print(f"[SUCCESS] Saved index and metadata to {db_path}")


def main():
    print(f"[INFO] Reading PDFs from: {DATA_PATH}")
    docs = gather_documents(DATA_PATH)
    print(f"[INFO] Found {len(docs)} chunks")
    if not docs:
        print("[WARN] No document chunks found. Put PDFs in the data/ directory and rerun.")
        return
    build_faiss_index(docs)


if __name__ == "__main__":
    main()

