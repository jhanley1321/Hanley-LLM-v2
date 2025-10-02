import streamlit as st

class ChatUI:
    """Streamlit chat interface - agnostic of LLM implementation"""

    def __init__(self, chat_handler):
        """
        Args:
            chat_handler: Any object with a .stream_chat(messages) method
        """
        self.chat_handler = chat_handler
        self._init_page()
        self._init_session_state()

    def _init_page(self):
        st.set_page_config(page_title="Ollama Chat", page_icon="🤖")
        st.title("🤖 Ollama Chat")

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
                
                # Use the injected chat handler
                for chunk in self.chat_handler.stream_chat(st.session_state.messages):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    def render_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            st.markdown("---")
            model_name = getattr(self.chat_handler, 'model', 'Unknown')
            st.markdown(f"**Model:** {model_name}")
            status = "🟢 Connected" if len(st.session_state.messages) else "⚪ Ready"
            st.markdown(f"**Status:** {status}")

    def run(self):
        self.render_chat_history()
        self.handle_user_input()
        self.render_sidebar()