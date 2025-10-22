import json
from datetime import datetime, timezone
from pathlib import Path


class ChatHistory:
    """
    Stores full chat turns (user prompt + model reply) per session.
    Each session is a single JSON file in chat_history_folder/.
    """

    def __init__(self, model_name: str = "llama2", chat_history_folder="chat_logs") -> None:
        self.model_name = model_name

        # Create a unique, UTC-safe session id and file
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.session_id = timestamp
        Path(chat_history_folder).mkdir(exist_ok=True)
        self.session_file = Path(chat_history_folder) / f"{timestamp}.json"

        base = {
            "_meta": {
                "session_id": self.session_id,
                "created_at": self.session_id,
                "model_name": self.model_name,
            },
            "turns": [],
        }
        self._write(base)

    # ----------------------------------------------------------------
    def _write(self, data: dict) -> None:
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    def _read(self) -> dict:
        with open(self.session_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------------------------------------------
    def add_turn(self, user_input: str, model_output: str) -> None:
        """
        Record a complete conversation turn:
        - user input
        - model output
        """
        data = self._read()
        data["turns"].append(
            {
                "input": user_input,
                "output": model_output,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
            }
        )
        self._write(data)