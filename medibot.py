# medibot.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from connect_memory_with_llm import Retriever  # import the class we created

st.set_page_config(page_title="Medical Chatbot (Local Ollama + FAISS)", page_icon="🏥", layout="centered")
st.title("🏥 Medical Chatbot (Local)")
st.caption("Local RAG: FAISS embeddings + Ollama (no cloud API keys)")

# cache the Retriever to avoid reloading models/index each interaction
@st.cache_resource
def get_retriever():
    return Retriever()

try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"Failed to initialize retriever: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for m in st.session_state.messages:
    role = m["role"]
    content = m["content"]
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask your medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Retrieving relevant docs and generating answer..."):
        try:
            result = retriever.answer(user_input, k=4)
            answer = result["answer"]
            sources = result["sources"]
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # show compact sources under the reply
            if sources:
                st.markdown("**Sources used:**")
                for s in sources:
                    meta = s["metadata"]
                    label = f"{meta.get('source','?')} (p.{meta.get('page','?')}) — score {s.get('score',0):.3f}"
                    st.markdown(f"- {label}")
else:
    st.info("Type a question to search your indexed PDFs and get an answer generated locally by Ollama.")
