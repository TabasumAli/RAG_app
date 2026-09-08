# 📄 PDF Knowledge Assistant — Simple RAG Demo

> A minimal **Retrieval-Augmented Generation (RAG)** app that lets you upload any PDF and ask it questions — with answers **grounded in your document**, not the model's memory.

Built as a hands-on project for the **"RAG Foundations & PDF Knowledge Assistant"** learning series.

---

## ✨ What It Does (The Use Case)

Upload a resume → ask *"What Python skills does this candidate have?"*
Upload a product manual → ask *"How do I reset the device?"*
Upload lecture notes → ask *"What were the 3 causes mentioned for X?"*

The app retrieves the most relevant chunks **from your PDF** and answers using only those — then shows you exactly which chunks it used as proof of grounding.

## 🧠 How RAG Works Here

```
INGESTION (offline, when you upload):
  PDF → Extract text → Chunk (400 words, 80 overlap)
      → Embed (all-MiniLM-L6-v2) → Store vectors in memory

QUERY (every question):
  Question → Embed → Cosine similarity → Top-3 chunks
          → Augmented prompt → openai/gpt-oss-120b → Grounded answer
```

This is **Standard (Naive) RAG** — the baseline architecture. No vector database, no framework magic: retrieval is one NumPy dot product, so the core idea stays visible.

## 🛠️ Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM | `openai/gpt-oss-120b` via **Groq** | Fast, free tier, OpenAI-compatible API |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) | Local, free, runs on CPU |
| Similarity | NumPy cosine similarity | Zero dependencies, fully transparent |
| PDF parsing | `pypdf` | Lightweight, pure Python |
| UI | **Streamlit** | Chat interface in ~100 lines |

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/TabasumAli/RAG_app.git
cd RAG_app
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Grab one at [console.groq.com](https://console.groq.com) — it takes a minute.

### 3. Run

```bash
streamlit run app.py
```

Open the browser → paste your API key in the sidebar → upload a PDF → ask questions.

## 🔐 Running on Streamlit Cloud (with secrets)

Don't type the key manually — add it as a secret:

**Manage app → Settings → Secrets:**

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Then replace the sidebar input in `app.py` with:

```python
api_key = st.secrets["GROQ_API_KEY"]
```

## 📁 Project Structure

```
RAG_app/
├── app.py              # The entire RAG pipeline (ingestion + retrieval + generation)
├── requirements.txt    # 5 dependencies, no vector DB needed
└── README.md
```

## 🎯 Why So Simple?

The goal of this project is **pedagogy, not production**. Every step of the RAG workflow — chunking, embedding, cosine search, prompt augmentation, grounded generation — is right there on the screen with no abstraction hiding it.

## 🔭 Upgrade Path

Once the baseline makes sense, natural next steps (each is one architecture from the series):

- [ ] **Conversational RAG** — remember chat history, rewrite follow-up questions
- [ ] **Corrective RAG** — grade retrieved chunks, re-retrieve or fall back to web search
- [ ] **ChromaDB/FAISS** — swap the in-memory store for a real vector database
- [ ] **Citations** — link answers to page numbers
- [ ] **Agentic RAG** — let an agent decide when/what to retrieve

## 📚 Key Takeaways (from the learning series)

- **The model's memory is not your knowledge base** — RAG grounds answers in *your* data.
- **Retrieval does the heavy lifting** — chunking and embeddings decide what the model ever sees. Garbage in, garbage out.
- **Guardrails make it trustworthy** — *"Answer only from the context. If you don't know, say so."*
- **Start simple, evolve deliberately** — Standard RAG first; corrective, speculative, and agentic variants are upgrades for specific limits.

## 📖 Learn More

This repo is the hands-on part of a learning guide covering: what RAG is, why it matters, key components, the workflow, architectures (Standard → Corrective → Speculative → Agentic), and RAG vs. fine-tuning.

---

⭐ If this helped you understand RAG, consider starring the repo!
