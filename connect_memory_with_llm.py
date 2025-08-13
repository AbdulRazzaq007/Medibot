# connect_memory_with_llm.py
import os
import pickle
import json
from typing import List, Dict

import numpy as np
import requests
from dotenv import load_dotenv
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # fallback for environments with PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

load_dotenv()

DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
INDEX_PICKLE = os.path.join(DB_PATH, "index.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct")

RETRIEVE_K = int(os.getenv("RETRIEVE_K", 4))
MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", 512))
TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", 0.2))


def _gather_docs_from_pdfs(data_dir: str):
    # fallback: if index.pkl missing, read PDFs and chunk (same as create_memory_for_llm)
    docs = []
    for path in sorted(os.listdir(data_dir)):
        if not path.lower().endswith(".pdf"):
            continue
        full = os.path.join(data_dir, path)
        reader = PdfReader(full)
        for i, p in enumerate(reader.pages):
            text = ""
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            text = " ".join(text.split())
            # simple chunking by characters
            start = 0
            while start < len(text):
                chunk = text[start:start+800].strip()
                if chunk:
                    docs.append({"page_content": chunk, "metadata": {"source": path, "page": i+1}})
                start += 800 - 120
    return docs


class Retriever:
    def __init__(self, index_pickle_path=INDEX_PICKLE, data_path=DATA_PATH):
        # load docs: prefer precomputed index.pkl
        if os.path.exists(index_pickle_path):
            with open(index_pickle_path, "rb") as f:
                data = pickle.load(f)
            self.docs = data.get("docs", [])
        else:
            print("[WARN] index.pkl not found — building doc chunks from PDFs (may be slower).")
            self.docs = _gather_docs_from_pdfs(data_path)

        if not self.docs:
            raise RuntimeError("No documents loaded. Put PDFs in data/ and run create_memory_for_llm.py locally.")

        # build TF-IDF matrix
        self.texts = [d["page_content"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=50000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> List[Dict]:
        q_vec = self.vectorizer.transform([query])
        # cosine similarity via linear_kernel
        sims = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(-sims)[:k]
        hits = []
        for idx in top_idx:
            score = float(sims[idx])
            doc = self.docs[idx]
            hits.append({"score": score, "page_content": doc["page_content"], "metadata": doc["metadata"]})
        return hits

    def _call_openrouter(self, messages):
        api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to environment or Streamlit secrets.")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False
        }
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Auth error from OpenRouter ({resp.status_code}). Check API key.")
            raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        # parse OpenAI-like response
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and isinstance(choice["message"], dict):
                return choice["message"].get("content", "").strip()
            if "text" in choice:
                return choice["text"].strip()
        # fallback
        return json.dumps(data)

    def answer(self, query: str, k: int = RETRIEVE_K) -> Dict:
        hits = self.retrieve(query, k=k)
        context_parts = []
        for h in hits:
            meta = h["metadata"]
            context_parts.append(f"[{meta.get('source','?')} p.{meta.get('page','?')}]\n{h['page_content']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

        system_msg = (
            "You are a careful medical assistant. Use ONLY the provided context to answer the user's question. "
            "If it is not in the context, say you don't know. Be concise and avoid hallucination."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        answer_text = self._call_openrouter(messages)
        return {"answer": answer_text, "sources": hits}


if __name__ == "__main__":
    r = Retriever()
    while True:
        q = input("Query (empty to quit): ").strip()
        if not q:
            break
        out = r.answer(q)
        print("\nANSWER:\n", out["answer"])
        print("\nSOURCES:")
        for s in out["sources"]:
            print(s["metadata"].get("source","?"), "p.", s["metadata"].get("page","?"), "score", s["score"])
            print(s["page_content"][:300].replace("\n"," "))
            print("-----")






