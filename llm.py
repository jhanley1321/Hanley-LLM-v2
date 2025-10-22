from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import Optional


class LLM:
    """Lowest layer; sends prompts to the model and returns text replies."""

    def __init__(self, model: str = "llama2", temperature: float = 0.2) -> None:
        self.model_name = model
        self.temperature = temperature
        self.client: Optional[ChatOllama] = None
        self.rag = None  # Will be set by Hanley
        self._init_model()

    def _init_model(self) -> None:
        self.client = ChatOllama(model=self.model_name, temperature=self.temperature)

    def set_rag(self, rag) -> None:
        """Attach RAG instance (called by Hanley during orchestration)."""
        self.rag = rag

    def send(self, message: str) -> str:
        """
        Send a message to the model.
        If RAG is enabled, retrieves context and uses send_rag internally.
        Otherwise, sends directly.
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized.")

        # Check if RAG is enabled
        if self.rag and self.rag.enabled:
            context = self.rag.build_context(message, top_k=3)
            return self._send_rag(message, context)
        else:
            return self._send_direct(message)

    def _send_direct(self, message: str) -> str:
        """Send directly to model (no RAG context)."""
        response = self.client.invoke([HumanMessage(content=message)])
        return response.content or ""

    def _send_rag(self, message: str, context: str) -> str:
        """Send with RAG context prepended."""
        composed = (
            "You must answer using only the provided context. "
            "If the answer is not in the context, say: Not in context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        )
        response = self.client.invoke([HumanMessage(content=composed)])
        return response.content or ""