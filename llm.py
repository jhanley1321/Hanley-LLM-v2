from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


class LLM:
    """
    Minimal LLM wrapper that:
      - Sends a chat message and returns the model's reply.
      - Keeps in-memory history and incrementally persists to a JSONL file.
    """

    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.2,
        log_path: Optional[str] = None,  # if provided, write each message incrementally
    ) -> None:
        self.model_name = model
        self.temperature = temperature
        self.client: Optional[ChatOllama] = None
        # Each entry: {"role": "user"|"assistant", "content": str, "timestamp": ISO-8601 str}
        self.history: List[Dict[str, str]] = []
        self.log_path: Optional[Path] = Path(log_path) if log_path else None
        self._init_model()
        if self.log_path:
            self._ensure_log_dir()

    def _init_model(self) -> None:
        # Assumes Ollama is running and the model is available: `ollama pull llama2`
        self.client = ChatOllama(model=self.model_name, temperature=self.temperature)

    def _now(self) -> str:
        # UTC ISO-8601 timestamp with 'Z'
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _ensure_log_dir(self) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not truncate existing logs; append-only.

    def _append_log(self, record: Dict[str, str]) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def send_message(self, message: str) -> str:
        """
        Send a user message to the LLM and return the assistant's reply.
        Stores both user message and assistant reply with timestamps, and
        appends each to the JSONL file if log_path is set.
        """
        if not self.client:
            raise RuntimeError("LLM client is not initialized.")

        user_rec = {"role": "user", "content": message, "timestamp": self._now()}
        self.history.append(user_rec)
        self._append_log(user_rec)

        # For now we only pass the latest user message to the model
        response = self.client.invoke([HumanMessage(content=message)])
        reply_text = response.content or ""

        asst_rec = {"role": "assistant", "content": reply_text, "timestamp": self._now()}
        self.history.append(asst_rec)
        self._append_log(asst_rec)

        return reply_text