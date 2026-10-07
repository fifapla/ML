from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class RAGEngine:
    def __init__(self, collection_name="document_collection"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.collection_name = collection_name
        self.vector_store = None

    def ingest_text(self, text_content):
        chunks = self.text_splitter.split_text(text_content)
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )
        return len(chunks)

    def query(self, query_str, k=3):
        if not self.vector_store:
            raise ValueError("No documents ingested yet.")
        results = self.vector_store.similarity_search(query_str, k=k)
        return [doc.page_content for doc in results]

if __name__ == "__main__":
    sample_doc = "Artificial Intelligence is transforming software engineering. RAG systems combine retrieval and generation."
    engine = RAGEngine()
    num_chunks = engine.ingest_text(sample_doc)
    print(f"Ingested {num_chunks} chunks.")
    res = engine.query("What is AI transforming?")
    print("Search Result:", res)
