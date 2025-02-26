import streamlit as st
from huggingface_hub import InferenceClient

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="fireworks-ai",
    api_key=st.secrets["HF_TOKEN"]
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """You are an AI assistant. For each query, first analyze the problem through logical steps.
        Present your thinking process between 'THINKING_START' and 'THINKING_END' markers, then provide the final answer after 'ANSWER_START'."""}
    ]

# Function to generate and parse responses
def generate_response(user_input):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Get API response
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=st.session_state.messages,
            max_tokens=500
        )
        
        # Extract response content
        full_response = completion.choices[0].message.content
        
        # Parse thinking process and answer
        thinking = ""
        answer = ""
        if "THINKING_START" in full_response:
            _, temp = full_response.split("THINKING_START", 1)
            thinking, answer = temp.split("ANSWER_START", 1)
            thinking = thinking.replace("THINKING_END", "").strip()
            answer = answer.strip()
        else:
            answer = full_response
        
        # Add parsed responses to session state
        if thinking:
            st.session_state.messages.append({"role": "assistant", "content": f"🧠 THINKING: {thinking}"})
        st.session_state.messages.append({"role": "assistant", "content": f"✅ ANSWER: {answer}"})
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# UI Configuration
st.set_page_config(page_title="DeepSeek Chat", layout="wide")
st.title("🤖 DeepSeek R1 Chat Assistant")

# Chat Interface
with st.container():
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                content = message["content"]
                
                if message["role"] == "user":
                    st.write(f"👤: {content}")
                else:
                    if content.startswith("🧠"):
                        st.markdown(f"*{content}*")
                    else:
                        st.success(content)

# User Input
user_input = st.chat_input("Type your message here...")
if user_input:
    generate_response(user_input)
    st.rerun()

# Conversation download logic
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False

# End Chat Button
col1, col2 = st.columns([0.2, 0.8])
with col1:
    if st.button("📥 End Chat & Generate Log"):
        st.session_state.download_ready = True

# Download functionality
if st.session_state.download_ready:
    chat_log = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_log.append(f"👤 User: {msg['content']}\n")
        elif msg["role"] == "assistant":
            chat_log.append(f"🤖 Assistant: {msg['content']}\n")
    
    txt_content = "".join(chat_log)
    
    st.download_button(
        label="⬇️ Download Conversation Log",
        data=txt_content,
        file_name="deepseek_chat_log.txt",
        mime="text/plain",
        key="download_button"
    )

# CSS for styling
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 20px;}
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {align-items: center;}
    .stDownloadButton button {width: 100%;}
</style>
""", unsafe_allow_html=True)
