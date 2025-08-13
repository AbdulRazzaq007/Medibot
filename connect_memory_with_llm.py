# connect_memory_with_llm.py
import os
import pickle
import time
import json
from typing import List, Dict

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", 4))
MAX_OUTPUT_TOKENS = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", 512))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.2))


class Retriever:
    def __init__(self, db_path=DB_PATH, embed_model=EMBED_MODEL, ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_MODEL):
        self.db_path = db_path
        self.embed_model_name = embed_model
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model

        # load FAISS index and docs
        idx_file = os.path.join(self.db_path, "index.faiss")
        pkl_file = os.path.join(self.db_path, "index.pkl")
        if not os.path.exists(idx_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"FAISS index or metadata not found in {self.db_path}. Run create_memory_for_llm.py first.")

        print(f"[INFO] Loading FAISS index from {idx_file}")
        self.index = faiss.read_index(idx_file)

        print(f"[INFO] Loading metadata from {pkl_file}")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        self.docs = data.get("docs", [])

        # sentence-transformers embedder
        print(f"[INFO] Loading embedding model: {self.embed_model_name}")
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], convert_to_numpy=True)
        emb = emb.astype(np.float32)
        # normalize
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        emb = emb / norm
        return emb

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> List[Dict]:
        q_emb = self._embed_query(query)
        D, I = self.index.search(q_emb, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            hits.append({"score": float(score), "page_content": doc["page_content"], "metadata": doc["metadata"]})
        return hits

    def _call_ollama(self, prompt: str, max_output_tokens: int = MAX_OUTPUT_TOKENS, temperature: float = TEMPERATURE) -> str:
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error contacting Ollama at {url}: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        # Ollama responses come in different formats; try several keys
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                return data["response"].strip()
            if "output" in data and isinstance(data["output"], list) and data["output"]:
                # output could be list of dicts with 'content'
                first = data["output"][0]
                if isinstance(first, dict) and "content" in first:
                    return first["content"].strip()
                if isinstance(first, dict) and "text" in first:
                    return first["text"].strip()
            # fallback: try choices
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                ch = data["choices"][0]
                if isinstance(ch, dict) and "message" in ch:
                    return ch["message"].get("content", "").strip()
                if isinstance(ch, dict) and "content" in ch:
                    return ch["content"].strip()
        # last resort:
        return json.dumps(data)

    def answer(self, query: str, k: int = RETRIEVE_K) -> Dict:
        # retrieve
        hits = self.retrieve(query, k=k)
        # build context
        context_parts = []
        for h in hits:
            meta = h["metadata"]
            context_parts.append(f"[{meta.get('source','unknown')} - p.{meta.get('page','?')}]\n{h['page_content']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

        system = "You are a careful medical assistant. Use ONLY the provided context to answer. If it's not in the context, say you don't know. Be concise."
        prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # call Ollama
        answer_text = self._call_ollama(prompt)
        return {"answer": answer_text, "sources": hits}


if __name__ == "__main__":
    # simple CLI demo
    r = Retriever()
    while True:
        try:
            q = input("\nWrite Query Here (empty to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        out = r.answer(q)
        print("\n--- ANSWER ---\n")
        print(out["answer"])
        print("\n--- SOURCES ---\n")
        for s in out["sources"]:
            print(s["metadata"].get("source", ""), "p.", s["metadata"].get("page", "?"), "score:", s["score"])
            print(s["page_content"][:400].replace("\n", " "))
            print("-----")





