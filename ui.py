import streamlit as st
from typing import Any, Protocol

class ChatHandler(Protocol):
    """
    Protocol defining the interface for chat handlers.
    
    Any class implementing stream_chat() can be used with ChatUI.
    """
    def stream_chat(self, messages: list) -> Any:
        """Stream chat responses given a list of messages."""
        ...

class ChatUI:
    """
    Streamlit chat interface - agnostic of LLM implementation.
    
    This class handles all UI logic for the chat interface and delegates
    actual LLM calls to an injected chat handler.
    """

    def __init__(self, chat_handler: ChatHandler):
        """
        Initialize the chat UI.
        
        Args:
            chat_handler: Any object with a .stream_chat(messages) method
        """
        self.chat_handler = chat_handler
        self._init_page()
        self._init_session_state()

    def _init_page(self) -> None:
        """Configure Streamlit page settings and title."""
        st.set_page_config(page_title="Ollama Chat", page_icon="🤖")
        st.title("🤖 Ollama Chat")

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state for message history."""
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_chat_history(self) -> None:
        """Display all previous messages in the chat."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self) -> None:
        """
        Handle new user input and generate assistant response.
        
        This method:
        1. Captures user input from chat input widget
        2. Adds user message to history and displays it
        3. Streams assistant response using the injected chat handler
        4. Adds assistant response to history
        """
        if prompt := st.chat_input("What would you like to know?"):
            # Add and display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Stream and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use the injected chat handler to get streaming response
                for chunk in self.chat_handler.stream_chat(st.session_state.messages):
                    full_response += chunk
                    # Show cursor while streaming
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove cursor when done
                message_placeholder.markdown(full_response)

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    def render_sidebar(self) -> None:
        """Render sidebar with settings and status information."""
        with st.sidebar:
            st.header("Settings")
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            
            st.markdown("---")
            
            # Display model info (if available from chat handler)
            model_name = getattr(self.chat_handler, 'model', 'Unknown')
            st.markdown(f"**Model:** {model_name}")
            
            # Display connection status
            status = "🟢 Connected" if len(st.session_state.messages) else "⚪ Ready"
            st.markdown(f"**Status:** {status}")

    def run(self) -> None:
        """
        Main run loop for the UI.
        
        This method orchestrates all UI components and should be called
        to start the Streamlit application.
        """
        self.render_chat_history()
        self.handle_user_input()
        self.render_sidebar()