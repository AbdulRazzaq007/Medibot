# medibot.py
import os
import streamlit as st
from dotenv import load_dotenv

# load local .env for local testing; on Streamlit Cloud use Secrets manager
load_dotenv()

from connect_memory_with_llm import Retriever

st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="centered")
st.title("🏥 Medical Chatbot")
st.caption("Search your indexed PDFs (FAISS) ")

# get OpenRouter key from env or Streamlit secrets
def get_openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY")
    try:
        if not key and "OPENROUTER_API_KEY" in st.secrets:
            key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return key

if not get_openrouter_key():
    st.error("Missing OpenRouter API key. Locally set OPENROUTER_API_KEY in .env or add OPENROUTER_API_KEY to Streamlit Secrets.")
    st.stop()

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

# show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# user input
user_msg = st.chat_input("Ask your medical question...")

if user_msg:
    # append and display user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("Retrieving and generating answer..."):
        try:
            result = retriever.answer(user_msg, k=int(os.getenv("RETRIEVE_K", 4)))
            answer = result["answer"]
            sources = result["sources"]
        except Exception as ex:
            st.error(f"Error: {ex}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {ex}"})
        else:
            # show assistant reply
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # compact sources listing
            if sources:
                st.markdown("**Sources used:**")
                for s in sources:
                    meta = s["metadata"]
                    src = meta.get("source", "unknown")
                    pg = meta.get("page", "?")
                    st.markdown(f"- {src} (p.{pg}) — score {s.get('score', 0):.4f}")

else:
    st.info("Type a question above — the app searches indexed PDFs and uses Mistral (OpenRouter) to answer.")
