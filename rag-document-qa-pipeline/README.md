# Domain-Specific RAG Document QA Pipeline
Retrieval-Augmented Generation (RAG) framework using LangChain, SentenceTransformers, and ChromaDB for local vector indexing and retrieval.

**Notes:**
- Needs internet access on first run (downloads the `all-MiniLM-L6-v2` sentence-transformer model from Hugging Face) and will not run in a fully offline/sandboxed environment.
- The `langchain.embeddings` / `langchain.vectorstores` import paths used here match older LangChain releases. Recent LangChain versions moved these into `langchain_community` (and `langchain_huggingface` for embeddings) - if imports fail, update them to match the LangChain version you install, or pin `langchain<0.1` to match this code as-is.
