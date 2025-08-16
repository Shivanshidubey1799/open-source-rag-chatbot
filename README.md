# Built a GUI RAG chatbot in Python using Gradio to query user documents (PDF/TXT/DOCX) with an intuitive chat interface.

Implemented an ingestion pipeline with chunking + overlap, SentenceTransformers embeddings (all-MiniLM-L6-v2), and FAISS vector search (fallback to sklearn).

Integrated Ollama for fully local, open-source LLM inference (e.g., llama3.1:8b), with runtime model switching via dropdown.

Designed retrieval + generation flow with Top-K passage ranking and inline source citations; side panel shows the exact retrieved chunks.

Packaged as a single-file app, easy to run in a conda environment; no external APIs—privacy-preserving and offline-friendly.

Tunable parameters (chunk size, overlap, Top-K) to balance accuracy vs. latency for different document sizes.open-source-rag-chatbot
