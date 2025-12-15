from datetime import datetime, timezone

class Chat:
    """
    Chat layer for managing conversation history.
    Stores messages in memory and optionally persists to disk.
    """

    def __init__(self):
        self.chat_log = []
        self.history = None

    def send(self, user_input: str) -> str:
        """
        Store user message in chat history.
        Returns the user input for further processing.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        self.chat_log.append({"role": "user", "content": user_input, "timestamp": timestamp})
        return user_input

    def add_assistant_response(self, response: str) -> None:
        """
        Store assistant response in chat history.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        self.chat_log.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        
        if self.history:
            user_msg = self.chat_log[-2]["content"] if len(self.chat_log) >= 2 else ""
            self.history.add_turn(user_msg, response)

    def get_messages(self):
        """
        Return the full chat log for sending to model.
        """
        return self.chat_log


class ChatOrchestrator:
    """
    Orchestrates the flow between Chat history and Model.
    Takes messages from Chat, sends to Model, returns response to Chat.
    """

    def __init__(self, chat: Chat, model_handler):
        """
        Initialize orchestrator with chat and model handler.
        
        Args:
            chat: Chat instance for managing conversation history
            model_handler: ModelHandler instance for sending messages to the model
        """
        self.chat = chat
        self.model_handler = model_handler

    def process_message(self, user_input: str) -> str:
        """
        Process a user message through the full pipeline:
        1. Store user message in chat history
        2. Send the user message to model
        3. Store model response in chat history
        4. Return model response
        
        Args:
            user_input: The user's message
            
        Returns:
            The model's response
        """
        self.chat.send(user_input)
        
        model_response = self.model_handler.send(user_input)
        
        self.chat.add_assistant_response(model_response)
        
        return model_response
