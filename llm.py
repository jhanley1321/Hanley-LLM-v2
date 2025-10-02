import requests
import json

class LLM:
    """Central LLM class for handling Ollama chat"""
    
    def __init__(self, model="llama2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.messages = []
    
    def chat(self, messages):
        """Send messages and return full response (non-streaming)."""
        try:
            response = requests.post(
                self.chat_url,
                json={"model": self.model, "messages": messages, "stream": False}
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, messages):
        """Send messages and yield response chunks (streaming)."""
        try:
            response = requests.post(
                self.chat_url,
                json={"model": self.model, "messages": messages, "stream": True},
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if "message" in json_response:
                        yield json_response["message"]["content"]
        except Exception as e:
            yield f"Error: {str(e)}"

    def run_cli(self):
        """Run an interactive CLI chat loop."""
        print("🤖 Ollama Chat (type 'exit' to quit)\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Add user msg
            self.messages.append({"role": "user", "content": user_input})

            # Stream reply
            print("Assistant: ", end="", flush=True)
            full_response = ""
            for chunk in self.stream_chat(self.messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            # Store response
            self.messages.append({"role": "assistant", "content": full_response})