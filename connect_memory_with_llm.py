# connect_memory_with_llm.py
"""
Retriever: loads FAISS index, does semantic search, and calls OpenRouter for generation.
"""

import os
import pickle
import json
from typing import List, Dict

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

# Config
DB_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # set locally or via Streamlit secrets
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", 4))
MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", 512))
TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", 0.2))


class Retriever:
    def __init__(self, db_path=DB_PATH, embed_model=EMBED_MODEL):
        self.db_path = db_path
        idx_file = os.path.join(self.db_path, "index.faiss")
        pkl_file = os.path.join(self.db_path, "index.pkl")

        if not os.path.exists(idx_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"FAISS index not found in {self.db_path}. Run create_memory_for_llm.py first.")

        print(f"[INFO] Loading FAISS index from {idx_file}")
        self.index = faiss.read_index(idx_file)

        print(f"[INFO] Loading metadata from {pkl_file}")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        self.docs = data.get("docs", [])

        print(f"[INFO] Loading embedder: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], convert_to_numpy=True)
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        emb = emb / norm
        return emb

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> List[Dict]:
        q_emb = self._embed_query(query)
        D, I = self.index.search(q_emb, k)
        hits = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            hits.append({"score": float(score), "page_content": doc["page_content"], "metadata": doc["metadata"]})
        return hits

    def _call_openrouter(self, messages: List[Dict], max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
        api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Set it in your environment or Streamlit secrets.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error when contacting OpenRouter: {e}")

        if resp.status_code != 200:
            # helpful error messages for common failures
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Authentication error from OpenRouter ({resp.status_code}). Check OPENROUTER_API_KEY and model access.")
            if resp.status_code == 404:
                raise RuntimeError(f"OpenRouter endpoint not found (404). URL: {OPENROUTER_URL}")
            # otherwise include body
            raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        # OpenRouter implements OpenAI-like chat-completions; parse safely
        try:
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                # prefer message.content
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content", "").strip()
                # fallback to text
                if "text" in choice:
                    return choice["text"].strip()
            # fallback: try other keys
            if isinstance(data, dict) and "output" in data:
                out = data["output"]
                if isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict) and "content" in first:
                        return first["content"].strip()
            return json.dumps(data)
        except Exception:
            return str(data)

    def answer(self, query: str, k: int = RETRIEVE_K) -> Dict:
        hits = self.retrieve(query, k=k)
        context_parts = []
        for h in hits:
            meta = h["metadata"] or {}
            label = f"[{meta.get('source', 'unknown')} p.{meta.get('page','?')}]"
            context_parts.append(f"{label}\n{h['page_content']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

        system_msg = (
            "You are a careful medical assistant. Use ONLY the provided context to answer the user's question. "
            "If the answer cannot be found in the context, say you don't know. Keep answers concise and do not hallucinate."
        )

        # Construct OpenRouter/OpenAI-style message list
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        answer_text = self._call_openrouter(messages)
        return {"answer": answer_text, "sources": hits}


# CLI test
if __name__ == "__main__":
    r = Retriever()
    print("[INFO] Retriever initialized. Type queries (empty to quit).")
    while True:
        q = input("\nQuery: ").strip()
        if not q:
            break
        out = r.answer(q)
        print("\n--- ANSWER ---\n")
        print(out["answer"])
        print("\n--- SOURCES ---\n")
        for s in out["sources"]:
            meta = s["metadata"]
            print(f"{meta.get('source','?')} p.{meta.get('page','?')} (score {s['score']:.4f})")
            print(s["page_content"][:400].replace("\n", " "))
            print("-----")





