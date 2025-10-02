# rag.py
import chromadb
from chromadb.utils import embedding_functions

class RAGPipeline:
    def __init__(self, persist_directory="./chroma_db", collection_name="docs"):
        # Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Default embedding function (SentenceTransformers under the hood)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def add_documents(self, docs: dict):
        """
        docs: { "id1": "doc text...", "id2": "another doc..." }
        """
        for doc_id, text in docs.items():
            self.collection.add(documents=[text], ids=[doc_id])

    def query(self, text, n_results=3):
        """Retrieve top docs for a query text."""
        results = self.collection.query(query_texts=[text], n_results=n_results)
        return results["documents"][0] if results and "documents" in results else []