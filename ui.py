import streamlit as st
import requests
import json

class ChatUI:
    """Streamlit chat interface for Ollama"""

    def __init__(self, model="llama2", ollama_url="http://localhost:11434/api/chat"):
        self.model = model
        self.ollama_url = ollama_url
        self._init_page()
        self._init_session_state()

    def _init_page(self):
        st.set_page_config(page_title="Minimal Ollama Chat", page_icon="🤖")
        st.title("🤖 Minimal Ollama Chat")

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        if prompt := st.chat_input("What would you like to know?"):
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    response = requests.post(
                        self.ollama_url,
                        json={
                            "model": self.model,
                            "messages": st.session_state.messages,
                            "stream": True
                        },
                        stream=True
                    )
                    for line in response.iter_lines():
                        if line:
                            json_response = json.loads(line)
                            if "message" in json_response:
                                chunk = json_response["message"]["content"]
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Error: {str(e)}. Make sure Ollama is running with: `ollama serve`"
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    def render_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            st.markdown("---")
            st.markdown(f"**Model:** {self.model}")
            status = "🟢 Connected" if len(st.session_state.messages) else "⚪ Ready"
            st.markdown(f"**Status:** {status}")

    def run(self):
        self.render_chat_history()
        self.handle_user_input()
        self.render_sidebar()