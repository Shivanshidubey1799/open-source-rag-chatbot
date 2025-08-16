#!/usr/bin/env python3
"""
GUI RAG Chatbot — open‑source stack (Gradio + FAISS + SentenceTransformers + Ollama)

Features
- Upload PDFs/TXT/DOCX; chunks, embeds, and indexes them (FAISS).
- Chat over your docs using Retrieval‑Augmented Generation.
- 100% local/open‑source by default (Ollama LLM + open embeddings).
- Shows which chunks were used (citations panel) and can export them.

Quickstart
1) Python 3.9–3.11 recommended.
2) Install Ollama (https://ollama.com) and pull a chat model, for example:
     ollama pull llama3.1:8b
     ollama pull mistral-nemo
   (You can change the model in the dropdown at runtime.)
3) pip install -r requirements.txt  (sample at bottom of this file comment)
4) python rag_chat_app.py  (this file)
5) Open the local Gradio URL printed in the terminal.

requirements.txt (suggested)
----
gradio>=4.42.0
faiss-cpu>=1.8.0.post1
sentence-transformers>=3.0.1
pypdf>=4.2.0
python-docx>=1.1.2
ollama>=0.3.1
numpy>=1.26.4
scikit-learn>=1.5.1
uvicorn>=0.30.5
----

Note: If FAISS install fails on your platform, replace FAISS with a simple
sklearn NearestNeighbors (code path included: set USE_FAISS=False below).
"""

from __future__ import annotations
import os
import io
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

# --- Config toggles ---
USE_FAISS = True  # set False to use sklearn NearestNeighbors instead of FAISS
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast, solid quality
DEFAULT_OLLAMA_MODEL = "gemma3:1b"
CHUNK_SIZE = 1000   # characters
CHUNK_OVERLAP = 200 # characters
TOP_K_DEFAULT = 5

# --- Imports that may be optional ---
import gradio as gr
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

try:
    import ollama  # pip install ollama
except Exception as _:
    ollama = None  # We'll raise a helpful error at runtime if missing

if USE_FAISS:
    try:
        import faiss  # pip install faiss-cpu
    except Exception as _:
        faiss = None
        USE_FAISS = False

if not USE_FAISS:
    from sklearn.neighbors import NearestNeighbors


# =====================
# Utility + RAG classes
# =====================

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str  # filename or label


class SimpleTextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        chunks: List[str] = []
        start = 0
        N = len(text)
        while start < N:
            end = min(start + self.chunk_size, N)
            chunk = text[start:end]
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            if end == N:
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks


class DocumentLoader:
    SUPPORTED = {".pdf", ".txt", ".docx"}

    @staticmethod
    def load_bytes(name: str, data: bytes) -> str:
        name_lower = name.lower()
        if name_lower.endswith(".pdf"):
            return DocumentLoader._read_pdf_bytes(data)
        elif name_lower.endswith(".txt"):
            return data.decode("utf-8", errors="ignore")
        elif name_lower.endswith(".docx"):
            with io.BytesIO(data) as buf:
                doc = docx.Document(buf)
                return "\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError(f"Unsupported file type for {name}. Supported: PDF/TXT/DOCX")

    @staticmethod
    def _read_pdf_bytes(data: bytes) -> str:
        with io.BytesIO(data) as buf:
            reader = PdfReader(buf)
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(texts)


class VectorStore:
    """A minimal vector store backed by FAISS or sklearn."""
    def __init__(self, dim: int, use_faiss: bool = USE_FAISS):
        self.dim = dim
        self.use_faiss = use_faiss
        self.embeddings: Optional[np.ndarray] = None  # for sklearn path
        self.ids: List[Tuple[str, int]] = []  # (doc_id, chunk_id)
        self.nn = None
        if self.use_faiss and faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def add(self, X: np.ndarray, ids: List[Tuple[str, int]]):
        X = _normalize(X)
        if self.use_faiss and self.index is not None:
            self.index.add(X)
        else:
            # store for sklearn
            if self.embeddings is None:
                self.embeddings = X
            else:
                self.embeddings = np.vstack([self.embeddings, X])
            # will fit NearestNeighbors on demand
        self.ids.extend(ids)

    def build(self):
        if not (self.use_faiss and self.index is not None):
            # build sklearn NN
            if self.embeddings is None or len(self.embeddings) == 0:
                raise ValueError("No embeddings to index.")
            self.nn = NearestNeighbors(metric="cosine")
            self.nn.fit(self.embeddings)

    def search(self, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        q = _normalize(q)
        if self.use_faiss and self.index is not None:
            sims, idxs = self.index.search(q, top_k)
            # FAISS returns inner product similarities; higher is better
        else:
            # cosine distance => convert to similarity = 1 - distance
            dists, idxs = self.nn.kneighbors(q, n_neighbors=top_k)
            sims = 1.0 - dists
        flat_idxs = idxs[0]
        return sims[0], [self.ids[i] for i in flat_idxs]


# ==========================
# RAG Orchestrator (in‑mem)
# ==========================

class RAGEngine:
    def __init__(self, embed_model_name: str = DEFAULT_EMBEDDING_MODEL, ollama_model: str = DEFAULT_OLLAMA_MODEL):
        self.embedder = SentenceTransformer(embed_model_name)
        # Normalize flag: we'll normalize ourselves
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.store = VectorStore(self.dim, use_faiss=USE_FAISS)
        self.splitter = SimpleTextSplitter()
        self.chunks: Dict[Tuple[str, int], Chunk] = {}
        self.doc_id_to_name: Dict[str, str] = {}
        self.ollama_model = ollama_model

    # --------- Building ---------
    def add_document(self, name: str, text: str) -> int:
        doc_id = sha1(name + str(len(text)))
        self.doc_id_to_name[doc_id] = name
        parts = self.splitter.split(text)
        metas: List[Chunk] = []
        ids: List[Tuple[str, int]] = []
        for i, t in enumerate(parts):
            c = Chunk(doc_id=doc_id, chunk_id=i, text=t, source=name)
            metas.append(c)
            ids.append((doc_id, i))
        if not parts:
            return 0
        embs = self.embedder.encode(parts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        self.store.add(embs, ids)
        for c in metas:
            self.chunks[(c.doc_id, c.chunk_id)] = c
        return len(parts)

    def finalize(self):
        self.store.build()

    # --------- Retrieval ---------
    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Tuple[Chunk, float]]:
        # Some embedding models (e.g., e5) like a prefix: "query: "
        qtext = query
        if "e5" in self.embedder.__class__.__name__.lower() or "e5" in self.embedder._modules.get("auto_model", None).__class__.__name__.lower():
            qtext = f"query: {query}"
        q = self.embedder.encode([qtext], convert_to_numpy=True)
        sims, id_list = self.store.search(q, top_k)
        out: List[Tuple[Chunk, float]] = []
        for sim, (doc_id, ck) in zip(sims, id_list):
            out.append((self.chunks[(doc_id, ck)], float(sim)))
        return out

    # --------- Generation ---------
    def answer(self, query: str, retrieved: List[Tuple[Chunk, float]],
               max_refs: int = 8, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        if ollama is None:
            raise RuntimeError("The 'ollama' package is not installed. Install Ollama and 'pip install ollama'.")
        if not retrieved:
            context = "(No relevant context found.)"
        else:
            blocks = []
            for i, (ck, sim) in enumerate(retrieved[:max_refs], start=1):
                label = f"[{i}] {ck.source}#chunk{ck.chunk_id} (sim={sim:.3f})"
                blocks.append(label + "\n" + ck.text)
            context = "\n\n---\n\n".join(blocks)

        sys_msg = system_prompt or (
            "You are a helpful assistant that answers strictly from the provided context. "
            "If the answer is not in the context, say you cannot find it. Be concise."
        )
        user_msg = (
            f"Answer the user question using only the CONTEXT. Include inline citations like [1], [2] "
            f"matching the sources.\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context}"
        )
        resp = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.2}
        )
        answer = resp["message"]["content"]
        return answer, context


# ==================
# Gradio application
# ==================

ENGINE: Optional[RAGEngine] = None
INDEX_BUILT = False


def ui_reset_state():
    global ENGINE, INDEX_BUILT
    ENGINE = RAGEngine()
    INDEX_BUILT = False


ui_reset_state()


def ingest_files(files: List[gr.File]) -> str:
    if not files:
        return "No files uploaded."
    total_chunks = 0
    for f in files:
        try:
            with open(f.name, "rb") as fh:
                data = fh.read()
            text = DocumentLoader.load_bytes(os.path.basename(f.name), data)
            n = ENGINE.add_document(os.path.basename(f.name), text)
            total_chunks += n
        except Exception as e:
            return f"Failed to process {f.name}: {e}"
    return f"Ingested {len(files)} file(s). Created {total_chunks} chunks."


def build_index() -> str:
    global INDEX_BUILT
    try:
        ENGINE.finalize()
        INDEX_BUILT = True
        return "Index built successfully. You can start chatting now."
    except Exception as e:
        return f"Index build failed: {e}"


def set_model(model_name: str) -> str:
    ENGINE.ollama_model = model_name.strip()
    return f"LLM set to: {ENGINE.ollama_model}"


def ask(chat_history: List[Tuple[str, str]], question: str, top_k: int) -> Tuple[List[Tuple[str, str]], str]:
    if not INDEX_BUILT:
        return chat_history, "Index is not built yet. Please click 'Build Index' first."
    if not question or not question.strip():
        return chat_history, "Please enter a question."
    try:
        hits = ENGINE.retrieve(question, top_k=top_k)
        answer, _ = ENGINE.answer(question, hits)
        chat_history = chat_history + [(question, answer)]
        # Prepare a simple citations string for the side panel
        cits = []
        for i, (ck, sim) in enumerate(hits, start=1):
            cits.append(f"[{i}] {ck.source}#chunk{ck.chunk_id}  sim={sim:.3f}\n{ck.text[:500]}{'...' if len(ck.text)>500 else ''}")
        citations_text = "\n\n\n".join(cits)
        return chat_history, citations_text
    except Exception as e:
        return chat_history, f"Error: {e}"


def clear_chat():
    return [], ""


with gr.Blocks(title="Open‑source RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Chat with Your Documents (Open‑source RAG)
    1. Upload PDF/TXT/DOCX → **Ingest**
    2. Click **Build Index**
    3. Ask questions in the chat. Answers include inline citations like [1], [2].
    """)

    with gr.Row():
        with gr.Column(scale=3):
            files = gr.File(label="Upload files", file_count="multiple", file_types=[".pdf", ".txt", ".docx"])
            ingest_btn = gr.Button("Ingest Files", variant="secondary")
            build_btn = gr.Button("Build Index", variant="primary")
            status = gr.Markdown("")

            model_dd = gr.Dropdown(
                label="Ollama model",
                choices=["gemma3:1b", "llama3.1:8b", "llama3.1:70b", "mistral-nemo", "qwen2.5:7b", "phi3:medium"],
                value=DEFAULT_OLLAMA_MODEL,
            )
            topk_slider = gr.Slider(1, 10, value=TOP_K_DEFAULT, step=1, label="Top‑K passages")

            set_model_btn = gr.Button("Set Model")
            reset_btn = gr.Button("Reset App")
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=420, show_copy_button=True)
            question = gr.Textbox(label="Ask a question about your documents", placeholder="e.g., Summarize section 3 …")
            send = gr.Button("Send", variant="primary")

        with gr.Column(scale=4):
            gr.Markdown("### Retrieved Context & Citations")
            citations = gr.Textbox(label="Most relevant chunks", lines=20)
            clear = gr.Button("Clear Chat")

    # Wire events
    ingest_btn.click(ingest_files, inputs=[files], outputs=[status])
    build_btn.click(build_index, outputs=[status])
    set_model_btn.click(set_model, inputs=[model_dd], outputs=[status])
    reset_btn.click(lambda: (ui_reset_state(), "App reset. Re‑ingest files and rebuild index.")[-1], outputs=[status])

    send.click(ask, inputs=[chatbot, question, topk_slider], outputs=[chatbot, citations])
    question.submit(ask, inputs=[chatbot, question, topk_slider], outputs=[chatbot, citations])
    clear.click(clear_chat, outputs=[chatbot, citations])


if __name__ == "__main__":
    # Run the Gradio app
    demo.launch()
