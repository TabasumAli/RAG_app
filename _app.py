"""
RAG Foundations — PDF Knowledge Assistant (Simple Demo)
========================================================
A minimal RAG app to demonstrate the core use case:
  Upload a PDF → Ask questions → Get grounded answers
  (with the retrieved chunks shown as evidence)

Model    : openai/gpt-oss-120b via Groq
Embeddings: all-MiniLM-L6-v2 (local, free, runs on CPU)
Vector DB : none needed — plain NumPy cosine similarity (keeps it simple)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- config
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400      # words per chunk
CHUNK_OVERLAP = 80    # words of overlap between chunks
TOP_K = 3             # how many chunks to retrieve


# ------------------------------------------------------------- functions
@st.cache_resource
def load_embedder():
    """Load the embedding model once (cached by Streamlit)."""
    return SentenceTransformer(EMBED_MODEL)


def load_pdf_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]  # drop tiny scraps


def retrieve(query: str, chunks: list[str], chunk_vecs: np.ndarray,
             embedder, top_k: int) -> list[str]:
    """Cosine-similarity search: return the top_k most relevant chunks."""
    q = embedder.encode([query], normalize_embeddings=True)
    scores = (chunk_vecs @ q.T).flatten()          # cosine (vectors normalized)
    top_idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def ask_groq(client: Groq, question: str, context_chunks: list[str]) -> str:
    context = "\n---\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using
ONLY the context below. If the answer is not in the context, say
"I don't know". Keep the answer concise.

Context:
{context}

Question: {question}"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ------------------------------------------------------------------ app
st.set_page_config(page_title="PDF Knowledge Assistant", page_icon="📄")
st.title("📄 PDF Knowledge Assistant — Simple RAG Demo")
st.caption("Upload a PDF, then ask it questions. "
           "Answers are grounded in YOUR document, not the model's memory.")

# --- sidebar: API key + PDF upload ---------------------------------
with st.sidebar:
    api_key = st.text_input("Groq API key", type="password",
                            help="Get one free at https://console.groq.com")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. PDF is split into overlapping chunks\n"
                "2. Chunks are embedded & stored in memory\n"
                "3. Your question is embedded → cosine similarity finds the top 3 chunks\n"
                "4. `openai/gpt-oss-120b` answers only from those chunks")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    if "chunks" not in st.session_state or st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading, chunking & embedding your PDF…"):
            embedder = load_embedder()
            text = load_pdf_text(uploaded)
            st.session_state.chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            st.session_state.chunk_vecs = embedder.encode(
                st.session_state.chunks, normalize_embeddings=True)
            st.session_state.file_name = uploaded.name
        st.success(f"Ready — {len(st.session_state.chunks)} chunks indexed from "
                   f"*{uploaded.name}*.")

    question = st.chat_input("Ask a question about your PDF…")
    if question:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()

        embedder = load_embedder()
        with st.spinner("Retrieving relevant chunks…"):
            hits = retrieve(question, st.session_state.chunks,
                            st.session_state.chunk_vecs, embedder, TOP_K)

        with st.spinner("Generating answer with openai/gpt-oss-120b…"):
            client = Groq(api_key=api_key)
            answer = ask_groq(client, question, hits)

        # answer
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)

        # proof of grounding — show what RAG retrieved
        with st.expander("🔍 Retrieved chunks (what RAG fed to the model)"):
            for i, chunk in enumerate(hits, 1):
                st.markdown(f"**Chunk {i}**")
                st.info(chunk)
else:
    st.info("👆 Upload a PDF to get started.")
    st.markdown("---")
    st.markdown("**Try it with:** a resume, a product manual, a research paper, "
                "lecture notes — anything you'd normally Ctrl-F through.")
