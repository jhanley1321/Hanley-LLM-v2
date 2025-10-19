import json
from datetime import datetime, timezone
from pathlib import Path


class ChatHistory:
    def __init__(self, model_name: str = "llama2") -> None:
        self.model_name = model_name

        # ✅ Use timezone-aware UTC for modern best practice and Windows-safe format
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.session_id = timestamp  # unique project identifier

        # ✅ Each project is one JSON file (no folders)
        Path("projects").mkdir(exist_ok=True)
        self.session_file = Path("projects") / f"{timestamp}.json"

        base = {
            self.model_name: [],
            "_meta": {
                "session_id": self.session_id,
                "created_at": self.session_id,
                "active_model": self.model_name,
                "history_order": [self.model_name],
            },
        }
        self._write(base)

    # ----------------------------------------------------------------
    def _write(self, data) -> None:
        """Write session content to disk."""
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    def _read(self):
        """Read the current project file."""
        with open(self.session_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        """Append a new message to the current model’s chat history."""
        data = self._read()
        model = self.model_name

        if model not in data:
            data[model] = []
            if model not in data["_meta"]["history_order"]:
                data["_meta"]["history_order"].append(model)
            data["_meta"]["active_model"] = model

        data[model].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
            }
        )
        self._write(data)