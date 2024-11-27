import streamlit as st
from model_loader import load_bloom_model
import torch
from streamlit_chat import message
import time

def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer = load_bloom_model()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_length' not in st.session_state:
        st.session_state.max_length = 100

def generate_response(prompt):
    """Generate response from the model"""
    system_prompt = """You are a helpful AI assistant.
    Your response should be point to point and clear , no extra things needed
    """
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
    inputs = st.session_state.tokenizer(full_prompt, return_tensors="pt").to(st.session_state.model.device)
    
    with torch.no_grad():   
        outputs = st.session_state.model.generate(
            **inputs,
            max_new_tokens=st.session_state.max_length,
            temperature=0.3,  # Reduced temperature for more focused responses
            do_sample=True,
            eos_token_id=st.session_state.tokenizer.eos_token_id,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    return response

def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append((user_input, True))
        st.session_state.thinking = True
        st.session_state.user_input = ""

def main():
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for chat interface
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding-bottom: 100px !important;
        }
        
        /* Messages container */
        .stChatMessageContent {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        
        .user-message {
            background-color: #e3effd;
        }
        
        /* Fixed input container at bottom */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 20px;
            z-index: 100;
            border-top: 1px solid #ddd;
            backdrop-filter: blur(10px);
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            padding-bottom: 100px;
            overflow-y: auto;
            height: calc(100vh - 100px);
        }
        
        /* Thinking animation */
        .thinking-msg {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
        }
        
        .thinking-msg .dots {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        
        /* Sidebar styling */
        .css-1544g2n {
            padding-top: 2rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Ensure messages don't go behind input */
        .element-container {
            padding-bottom: 50px;
        }
        
        /* Style scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("🤖 Chatbot Settings")
        st.markdown("---")
        
        # Model settings
        st.subheader("Model Parameters")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused and deterministic."
        )
        
        st.session_state.max_length = st.slider(
            "Maximum Length",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Maximum length of the generated response."
        )
        
        # Clear chat button
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot is powered by the BLOOM language model.
        - Use the settings above to adjust the response generation
        - Clear the chat history using the button above
        - Type your message in the input field at the bottom
        """)

    # Main chat interface
    st.title("💬 AI Chat Assistant")
    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    # Input container at bottom
    input_container = st.container()
    
    # Display messages in chat container
    with chat_container:
        for i, (msg, is_user) in enumerate(st.session_state.messages):
            message(msg, is_user=is_user, key=str(i))
        
        if st.session_state.thinking:
            with st.chat_message("assistant"):
                st.markdown("""
                    <div class="thinking-msg">
                        <span>Thinking</span>
                        <span class="dots">...</span>
                    </div>
                """, unsafe_allow_html=True)
                
                response = generate_response(st.session_state.messages[-1][0])
                st.session_state.messages.append((response, False))
                st.session_state.thinking = False
                st.rerun()
    
    # Fixed input field at bottom
    with input_container:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.text_input(
            "Message",
            key="user_input",
            placeholder="Type your message here...",
            on_change=on_input_change,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 