from typing import Protocol, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


class VectorStore(Protocol):
    """
    Protocol defining the interface for vector stores.
    
    Any vector store implementation (LangChain, LlamaIndex, raw Chroma, etc.)
    must implement these methods to be compatible with the RAG pipeline.
    """
    
    def add_documents(self, docs: Dict[str, str]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            docs: Dictionary mapping document IDs to document text
        """
        ...
    
    def query(self, text: str, n_results: int = 3) -> List[str]:
        """
        Query the vector store for similar documents.
        
        Args:
            text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            List of document texts most similar to the query
        """
        ...


class LangChainVectorStore:
    """
    LangChain-based vector store implementation using Chroma and Ollama embeddings.
    
    This adapter wraps LangChain's Chroma vectorstore to conform to our
    VectorStore protocol, allowing easy swapping with other implementations.
    """
    
    def __init__(
        self, 
        collection_name: str = "docs",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "mxbai-embed-large"
    ):
        """
        Initialize the LangChain vector store.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the vector database
            embedding_model: Ollama embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize or load Chroma vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(self, docs: Dict[str, str]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            docs: Dictionary mapping document IDs to document text
        """
        # Convert dict to lists for LangChain
        texts = list(docs.values())
        metadatas = [{"id": doc_id} for doc_id in docs.keys()]
        
        # Add to vectorstore
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
    
    def query(self, text: str, n_results: int = 3) -> List[str]:
        """
        Query the vector store for similar documents.
        
        Args:
            text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            List of document texts most similar to the query
        """
        # Perform similarity search
        results = self.vectorstore.similarity_search(text, k=n_results)
        
        # Extract just the text content
        return [doc.page_content for doc in results]