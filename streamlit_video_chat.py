import streamlit as st
import requests
import json
import re
from datetime import datetime
import time
from urllib.parse import urlparse, parse_qs

# Configure page
st.set_page_config(
    page_title="VideoChat - AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .app-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #ff4b4b, #ff8f70);
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.2);
    }
    
    .app-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
    }
    
    .chat-container {
        border-radius: 10px;
        background-color: rgba(247, 247, 247, 0.7);
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .video-info {
        border-radius: 10px;
        background-color: rgba(247, 247, 247, 0.7);
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .status-badge-success {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .status-badge-pending {
        background-color: #ffc107;
        color: white;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .status-badge-error {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .suggestion-btn {
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        background-color: white;
        padding: 8px 15px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .suggestion-btn:hover {
        background-color: #f0f0f0;
        border-color: #c0c0c0;
    }
    
    .empty-chat {
        text-align: center;
        padding: 30px;
        color: #666;
    }
    
    .empty-chat-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        color: #ddd;
    }
    
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Our API Endpoint Configuration
API_BASE_URL = "https://youtuberag-l55k.onrender.com"
PROCESS_VIDEO_ENDPOINT = f"{API_BASE_URL}/process_video"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# Utility functions
def extract_video_id(video_url):
    """Extract the video ID from a YouTube URL."""
    parsed_url = urlparse(video_url)
    
    # Handle youtube.com URLs
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [''])[0]
    
    # Handle youtu.be URLs
    elif parsed_url.netloc == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    return ""


def process_youtube_video(video_url, language="en"):
    """Process a YouTube video through the API."""
    try:
        response = requests.post(
            PROCESS_VIDEO_ENDPOINT,
            json={"video_url": video_url, "language": language}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def chat_with_video(video_id, question):
    """Send a question to the API about a processed video."""
    try:
        start_time = time.time()
        response = requests.post(
            CHAT_ENDPOINT,
            json={"video_id": video_id, "question": question}
        )
        response.raise_for_status()
        end_time = time.time()
        response_data = response.json()
        response_data["latency"] = round(end_time - start_time, 2)
        return response_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting answer: {str(e)}")
        return None

# session state
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_status" not in st.session_state:
    st.session_state.video_status = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

# App Header
st.markdown("""
<div class="app-header">
    <h1>YouTube Chat</h1>
    <p><strong>Transform  YouTube video into an interactive conversation with AI</strong></p>
</div>
""", unsafe_allow_html=True)

# Main layout with two columns
col1, col2 = st.columns([1, 2])

# Left Column : Video Processing
with col1:
    st.markdown("### Step 1: Process a Video")
    
    with st.container(border=True):
        video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        language_options = {
            "en": "English", 
            "es": "Spanish",
            "ar": "Arabic",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean"
           
        }
        language = st.selectbox("Transcript Language", options=list(language_options.keys()), 
                               format_func=lambda x: language_options[x], index=0)
        
        process_btn = st.button("Process Video", type="primary", use_container_width=True)
        
        if process_btn and video_url:
            # Clear chat history when processing a new video
            st.session_state.chat_history = []
            
            with st.status("Processing video...", expanded=True) as status:
                st.write("Extracting transcript...")
                video_id = extract_video_id(video_url)
                
                if not video_id:
                    st.error("Invalid YouTube URL. Please check the URL and try again.")
                else:
                    st.write("Creating embeddings...")
                    result = process_youtube_video(video_url, language)
                    
                    if result and result.get("status") == "completed":
                        st.session_state.video_id = result.get("video_id")
                        st.session_state.video_status = "completed"
                        st.session_state.processing_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        status.update(label="✅ Video processed successfully!", state="complete", expanded=False)
                    else:
                        st.session_state.video_status = "failed"
                        status.update(label="❌ Video processing failed", state="error", expanded=True)
                        if result:
                            st.error(result.get("message", "Unknown error occurred"))
        
    # Show video information if processed
    if st.session_state.video_id:
        st.markdown("### Video Information")
        
        with st.container(border=True):
            video_id = st.session_state.video_id
            
            # Display embedded YouTube video player instead of thumbnail
            youtube_embed_html = f'''
            <iframe width="100%" height="200" src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; 
            gyroscope; picture-in-picture" allowfullscreen></iframe>
            '''
            st.components.v1.html(youtube_embed_html, height=315)
            
            # Display video status
            if st.session_state.video_status == "completed":
                st.markdown(f"<div class='status-badge-success'>✅ Ready for chat</div>", unsafe_allow_html=True)
            elif st.session_state.video_status == "failed":
                st.markdown(f"<div class='status-badge-error'>❌ Processing failed</div>", unsafe_allow_html=True)
            
            # Video metadata
            st.markdown(f"**Video ID:** `{video_id}`")
            st.markdown(f"**Processed:** {st.session_state.processing_time}")
            
            # YouTube link
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            st.link_button("Open in YouTube", url=youtube_url)

            
            # Tips
            with st.expander("💡 Tips for better results"):
                st.markdown("""
                - Ask specific questions about the video content
                - For long videos, ask about specific sections
                - Try reformulating your question if the answer isn't helpful
                """)

# Right Column : Chat Interface
with col2:
    st.markdown("### Step 2: Chat with the Video")
    
    # Show the chat interface only if a video is processed
    if st.session_state.video_id and st.session_state.video_status == "completed":
        chat_container = st.container(height=500, border=True)
        
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="empty-chat">
                    <div class="empty-chat-icon">💬</div>
                    <h3>Start the conversation</h3>
                    <p>Ask a question about the video to get a detailed answer.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "latency" in message and message["role"] == "assistant":
                            st.caption(f"Response time: {message['latency']}s")
        
        st.markdown("**Try asking:**")
        
        # Suggested questions

        suggested_questions = [
            "What is this video about?",
            "What are the main points discussed in this video?",
            "Can you summarize this video in 3 bullet points?",
            "What examples or evidence were used?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}"):
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    # Get AI response
                    with st.status("Analyzing video content...", expanded=True) as status:
                        response = chat_with_video(st.session_state.video_id, question)
                        if response:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "latency": response["latency"]
                            })
                            status.update(label="✅ Answer ready!", state="complete", expanded=False)
                        else:
                            status.update(label="❌ Failed to get answer", state="error", expanded=True)
                    
                    st.rerun()
        
        with st.container():
            with st.form(key="chat_form", clear_on_submit=True):
                user_question = st.text_area(
                    "Ask me anything about this video...",
                    height=80,
                    max_chars=500,
                    key="user_input"
                )
                submit_button = st.form_submit_button("Send", use_container_width=True)
                
                if submit_button and user_question:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_question
                    })
                    
                    with st.status("Analyzing video content ...", expanded=True) as status:
                        response = chat_with_video(st.session_state.video_id, user_question)
                        if response:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "latency": response["latency"]
                            })
                            status.update(label="✅ Answer ready!", state="complete", expanded=False)
                        else:
                            status.update(label="❌ Failed to get answer", state="error", expanded=True)
                    
                    st.rerun()
                
    else:
        st.info("👆 First, process a YouTube video to start chatting.")
        st.markdown("""
        1. Paste a YouTube URL in the left panel
        2. Select the transcript language
        3. Click "Process Video"
        4. Once processing is complete, you can start asking questions
        """)


# Add download option for chat history
if st.session_state.chat_history:
    st.sidebar.title("Export Options")
    
    # Download chat as JSON
    json_data = json.dumps(st.session_state.chat_history, indent=2)
    st.sidebar.download_button(
        label="Download Chat (JSON)",
        data=json_data,
        file_name=f"videochat_{st.session_state.video_id}.json",
        mime="application/json"
    )
    
    # Download chat as Markdown
    markdown_content = f"# Video Chat: {st.session_state.video_id}\n\n"
    markdown_content += f"Date: {st.session_state.processing_time}\n\n"
    
    for message in st.session_state.chat_history:
        role = "User" if message["role"] == "user" else "AI Assistant"
        markdown_content += f"## {role}:\n\n{message['content']}\n\n"
    
    st.sidebar.download_button(
        label="Download Chat (Markdown)",
        data=markdown_content,
        file_name=f"videochat_{st.session_state.video_id}.md",
        mime="text/markdown"
    )
    
    # Clear chat history button
    if st.sidebar.button("Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()