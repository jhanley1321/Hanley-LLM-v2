from datetime import datetime, timezone

class Chat:
    """
    Chat layer between user and model.
    Stores transient memory and passes messages to the attached model.
    """

    def __init__(self):
        self.chat_log = []
        self._model = None

    def set_model(self, model):
        """Attach any backend model that implements send(message:str)->str."""
        self._model = model

    def send(self, user_input: str) -> str:
        """Accept user input, forward to model, store both messages in memory."""
        if not self._model:
            raise RuntimeError("No model attached to Chat layer. Use set_model() first.")

        # user message
        timestamp = datetime.now(timezone.utc).isoformat()
        self.chat_log.append({"role": "user", "content": user_input, "timestamp": timestamp})

        # model reply
        reply = self._model.send(user_input)
        self.chat_log.append({"role": "assistant", "content": reply,
                              "timestamp": datetime.now(timezone.utc).isoformat()})
        return reply