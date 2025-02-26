import streamlit as st
import requests
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="DeepSeek Chatbot",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for chat history
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

def generate_chat_file():
    """Generate text file content from chat history"""
    file_content = "Chat History\n\n"
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        file_content += f"{role}: {msg['content']}\n\n"
    return file_content

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password")
    
    if st.button("Export Conversation"):
        if len(st.session_state.chat_history) == 0:
            st.warning("No conversation to export")
        else:
            chat_file = generate_chat_file()
            st.download_button(
                label="Download Chat",
                data=chat_file,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# Main chat interface
st.title("DeepSeek Cognitive Assistant")
st.caption("A structured problem-solving chatbot with transparent reasoning")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant" and "Thinking:" in content:
            parts = content.split("Answer:", 1)
            if len(parts) == 2:
                st.markdown(f"**Thinking Process**\n{parts[0].replace('Thinking:', '').strip()}")
                st.divider()
                st.markdown(f"**Solution**\n{parts[1].split('Task:')[0].strip()}")
                if "Task:" in content:
                    st.divider()
                    st.markdown(f"**Next Steps**\n{content.split('Task:')[1].strip()}")
            else:
                st.write(content)
        else:
            st.write(content)

# User input handling
if prompt := st.chat_input("Enter your problem or question..."):
    if not hf_token:
        st.warning("Please add your Hugging Face token in the sidebar")
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
        # Get AI response
        response = query_hf({"inputs": messages}, hf_token)
        
        if isinstance(response, list) and 'generated_text' in response[0]:
            full_response = response[0]['generated_text']
        else:
            full_response = "Error: Unable to generate response. Please try again."
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.rerun()
    
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        st.session_state.chat_history.pop()  # Remove last user message if failed
