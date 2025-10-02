import requests
import json
from typing import List, Dict, Generator

class LLM:
    """
    Central LLM class for handling Ollama chat.
    
    This class provides methods for both streaming and non-streaming chat,
    and includes a built-in CLI interface for testing.
    """
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize the LLM client.
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.messages: List[Dict[str, str]] = []  # Chat history for CLI mode
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages and return full response (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Complete response string from the model
        """
        try:
            response = requests.post(
                self.chat_url,
                json={"model": self.model, "messages": messages, "stream": False}
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Send messages and yield response chunks (streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Yields:
            Response chunks as they arrive from the model
        """
        try:
            response = requests.post(
                self.chat_url,
                json={"model": self.model, "messages": messages, "stream": True},
                stream=True,
            )
            response.raise_for_status()
            
            # Parse streaming JSON responses
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if "message" in json_response:
                        yield json_response["message"]["content"]
        except Exception as e:
            yield f"Error: {str(e)}"

    def run_cli(self) -> None:
        """
        Run an interactive CLI chat loop.
        
        This method provides a simple command-line interface for testing
        the LLM. Type 'exit' or 'quit' to end the session.
        """
        print("🤖 Ollama Chat (type 'exit' to quit)\n")

        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})

            # Stream and display assistant response
            print("Assistant: ", end="", flush=True)
            full_response = ""
            for chunk in self.stream_chat(self.messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": full_response})