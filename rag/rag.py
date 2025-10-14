import csv
from typing import Dict, List, Optional, Any


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    """

    def __init__(self, vector_store: Any, csv_path: str):
        self.vector_store = vector_store
        self.csv_path = csv_path
        self._index_built: bool = False

    def load_csv_documents(self) -> Dict[str, str]:
        """Load reviews from CSV into {doc_id: review_text} dict."""
        docs: Dict[str, str] = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                review: Optional[str] = row.get("review") or row.get("text") or str(row)
                if review:
                    docs[f"review_{i}"] = review
        return docs

    def has_existing_index(self) -> bool:
        """
        Check if the vector store already contains indexed documents.
        
        Returns:
            True if the store has data, False otherwise
        """
        try:
            results = self.vector_store.query("ping", n_results=1)
            return len(results) > 0
        except Exception:
            # If query fails (e.g., collection doesn't exist), assume empty
            return False

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build (or reuse) vector store index.
        
        Only checks persistence on first call. After that, uses in-memory flag.
        
        Args:
            force_rebuild: if True, re-index from scratch even if already built
        """
  
        # Persistent check: does the store already have data on disk?
        if self.has_existing_index() and not force_rebuild:
            print("🔄 Using existing persisted index. Skipping rebuild.")
            self._index_built = True
            return

        # Build from scratch
        docs: Dict[str, str] = self.load_csv_documents()
        print(f"📥 Indexing {len(docs)} docs from {self.csv_path} ...")
        self.vector_store.add_documents(docs)
        print("✅ Index build complete!")
        self._index_built = True

    def query(self, question: str, n_results: int = 3) -> List[str]:
        """Query the vector store for similar docs."""
        return self.vector_store.query(question, n_results)