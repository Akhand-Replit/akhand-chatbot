import streamlit as st
from huggingface_hub import InferenceClient
import time
import textwrap
import os
from datetime import datetime

# Set up page configuration
st.set_page_config(
    page_title="DeepSeek R1 Chat Assistant",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatInput {position: fixed; bottom: 3rem; width: 100%;}
    .stDownloadButton {display: block; margin: 0 auto;}
    .thinking-bubble {
        background: #f0f2f6;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False
if "text_buffer" not in st.session_state:
    st.session_state.text_buffer = ""

# Initialize HF client
@st.cache_resource
def get_client():
    return InferenceClient(token=st.secrets["HF_TOKEN"])

client = get_client()

# System prompt for structured thinking
SYSTEM_PROMPT = """You are a helpful assistant that follows this structure:
[THINKING]
- Analyze user's query
- Break down problem into components
- Identify solution approach
[/THINKING]

[ANSWER]
Provide final comprehensive answer here
[/ANSWER]

Always maintain this structure and keep conversations professional yet friendly."""

def generate_conversation(user_input):
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"
    
    response = client.text_generation(
        prompt=prompt,
        model="deepseek-ai/DeepSeek-R1",
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        return_full_text=False
    )
    
    return response.strip()

def format_response(response):
    # Split into thinking and answer sections
    thinking = ""
    answer = ""
    
    if "[THINKING]" in response and "[/THINKING]" in response:
        thinking = response.split("[THINKING]")[1].split("[/THINKING]")[0].strip()
        answer = response.split("[/THINKING]")[1].split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()
    else:
        answer = response
    
    return thinking, answer

# Chat interface
st.title("DeepSeek R1 Assistant")
st.caption("An AI assistant that shows its thinking process")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if not st.session_state.conversation_ended:
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.text_buffer += f"User: {prompt}\n\n"
        
        # Generate response
        with st.spinner("Analyzing..."):
            raw_response = generate_conversation(prompt)
            thinking, answer = format_response(raw_response)
        
        # Display thinking process
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                if thinking:
                    wrapped_thinking = textwrap.fill(thinking, width=80)
                    thinking_placeholder.markdown(f"<div class='thinking-bubble'>💭 **Thinking Process**\n\n{wrapped_thinking}</div>", 
                                                 unsafe_allow_html=True)
                    st.session_state.text_buffer += f"Assistant Thinking:\n{thinking}\n\n"
                    time.sleep(2)  # Simulate processing time
                
                # Display final answer
                answer_placeholder = st.empty()
                wrapped_answer = textwrap.fill(answer, width=80)
                answer_placeholder.markdown(wrapped_answer)
                st.session_state.text_buffer += f"Assistant Answer:\n{answer}\n\n"
                
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Download functionality
if st.session_state.messages and not st.session_state.conversation_ended:
    if st.button("End Conversation"):
        st.session_state.conversation_ended = True
        st.rerun()

if st.session_state.conversation_ended:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    
    st.download_button(
        label="Download Conversation",
        data=st.session_state.text_buffer,
        file_name=filename,
        mime="text/plain"
    )
    
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_ended = False
        st.session_state.text_buffer = ""
        st.rerun()
