import streamlit as st
import requests
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="DeepSeek Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# System prompt template
SYSTEM_PROMPT = """You are an AI assistant that helps solve complex problems. Always follow these steps:
1. Thinking: Analyze the problem carefully and outline your thought process
2. Answer: Provide a clear, structured solution to the problem
3. Task: Suggest next steps or ask clarifying questions to continue the conversation

Format your response exactly like this:
Thinking: [your analytical process here]
Answer: [your solution here]
Task: [suggested next steps here]"""

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-r1"

def query_hf(payload, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def parse_response(response):
    # Use regex to extract sections
    sections = {
        'thinking': re.search(r'Thinking:\s*(.*?)(?=\nAnswer:|\nTask:|$)', response, re.DOTALL),
        'answer': re.search(r'Answer:\s*(.*?)(?=\nTask:|$)', response, re.DOTALL),
        'task': re.search(r'Task:\s*(.*)', response, re.DOTALL)
    }
    
    parsed = {k: v.group(1).strip() if v else None for k, v in sections.items()}
    return parsed

def generate_chat_file():
    """Generate text file content from chat history"""
    file_content = "DeepSeek Chat History\n\n"
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("Thinking:", "").replace("Answer:", "").replace("Task:", "")
        file_content += f"{role}: {content}\n\n"
    return file_content

# Sidebar for settings and export
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=st.secrets.get("HF_TOKEN", "")  # Get from secrets if available
    )
    
    if st.button("📥 Export Conversation", use_container_width=True):
        if not st.session_state.chat_history:
            st.warning("No conversation to export")
        else:
            chat_file = generate_chat_file()
            st.download_button(
                label="Download Chat History",
                data=chat_file,
                file_name=f"deepseek_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# Main chat interface
st.title("🧠 DeepSeek Cognitive Assistant")
st.caption("A structured problem-solving assistant with transparent reasoning")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            parsed = parse_response(message["content"])
            
            # Display thinking process
            if parsed['thinking']:
                with st.status("💭 Thinking Process", expanded=False):
                    st.write(parsed['thinking'])
            
            # Display answer
            if parsed['answer']:
                st.markdown("#### 📝 Solution")
                st.write(parsed['answer'])
            
            # Display next steps
            if parsed['task']:
                st.divider()
                st.markdown("#### 🔜 Next Steps")
                st.write(parsed['task'])
        else:
            st.write(message["content"])

# User input handling
if prompt := st.chat_input("Enter your problem or question..."):
    if not hf_token:
        st.warning("🔑 Please add your Hugging Face token in the sidebar")
        st.stop()

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Construct conversation prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[{"role": msg["role"], "content": msg["content"]} 
          for msg in st.session_state.chat_history]
    ]
    
    try:
        with st.spinner("🤖 Processing..."):
            response = query_hf({"inputs": messages}, hf_token)
            
            if isinstance(response, dict) and 'error' in response:
                st.error(f"API Error: {response['error']}")
                st.session_state.chat_history.pop()
            else:
                # Handle different response formats
                if isinstance(response, list):
                    full_response = response[0].get('generated_text', '')
                else:
                    full_response = response.get('generated_text', '')
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                st.rerun()
                
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.session_state.chat_history.pop()
