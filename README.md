# YouTube RAG - Video Chat Assistant 🎥💬

A powerful application that transforms YouTube videos into interactive conversations using Simple  Retrieval-Augmented Generation RAG Architecture. Ask questions about YouTube video and get precise answers based on its content.

## Features

- **Video Processing**: Extract and analyze content from  YouTube video with available transcripts
- **Multi-Language Support**: Process videos with transcripts in multiple languages
- **Simple RAG Pipeline**: Accurately retrieve and contextualize relevant information from videos
- **Interactive Chat**: Ask questions about the video content and receive detailed answers
- **Conversation History**: Save and export your conversation history in JSON or Markdown format
- **Modern UI**: Clean and intuitive Streamlit interface

##  Technical Overview

The project consists of two main components:

1. **FastAPI Backend**: The core RAG system that processes videos and answers questions
2. **Streamlit Frontend**: User-friendly interface for interacting with the RAG system

### RAG Architecture

The application uses a the Simple RAG  pipeline:

1. **Video Processing**:
   - Extract transcript from YouTube video
   - Split transcript into meaningful chunks
   - Generate embeddings using Cohere's embed-english-v3.0 model
   - Store embeddings in Pinecone vector database

2. **Question Answering**:
   - Process user question through MultiQueryRetriever
   - Retrieve most relevant transcript chunks from Pinecone
   - Generate comprehensive answer using Google's Gemini 2.5 Flash model

##  Installation

### Prerequisites

- Python 3.10+
- API keys for:
  - Pinecone
  - Cohere
  - Google AI (Gemini)

### Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv youtube_rag_env
   # On Windows
   .\youtube_rag_env\Scripts\activate
   # On Unix or MacOS
   source youtube_rag_env/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys:
   ```
   GOOGLE_API_KEY="your_google_api_key"
   COHERE_API_KEY="your_cohere_api_key"
   PINECONE_API_KEY="your_pinecone_api_key"
   ```

## Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn youtube_rag_api:app --reload
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   streamlit run streamlit_video_chat.py
   ```

3. Open your browser and navigate to `http://localhost:8501`


## API Endpoints

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/` | GET | Health check | None | Status message |
| `/process_video` | POST | Process video transcript | `video_url`, `language` | Processing status |
| `/chat` | POST | Answer questions about a video | `video_id`, `question` | Answer with sources |

You can find the API documentation [here](https://youtuberag-l55k.onrender.com/docs).

## Usage

1. Paste a YouTube URL in the input field
2. Select the language of the video's transcript
3. Click "Process Video" to extract and analyze the content
4. Once processing is complete, ask questions about the video in the chat interface
5. Use the suggested questions or type your own
6. Download your conversation history as JSON or Markdown from the sidebar

## Project Structure

- `youtube_rag_api.py` - FastAPI backend implementing the RAG system
- `streamlit_video_chat.py` - Streamlit frontend for user interaction
- `requirements.txt` - Required Python packages
- `notebooks/YouTubeRAG.ipynb` - Development notebook with detailed explanations
- `img/technical_architecture.png` - Technical architecture diagram

## Environment Variables

The application requires the following environment variables:

- `GOOGLE_API_KEY` - API key for Google's Gemini model
- `COHERE_API_KEY` - API key for Cohere's embedding model
- `PINECONE_API_KEY` - API key for Pinecone vector database

## Limitations

- Only works with YouTube videos that have available transcripts
- Quality of answers depends on the quality and accuracy of video transcripts
- Processing very long videos may take more time and resources

## Technologies Used

- **Backend**: FastAPI, LangChain, Pinecone
- **Frontend**: Streamlit
- **AI/ML**: 
  - Cohere Embeddings (embed-english-v3.0)
  - Google Gemini 2.5 Flash
- **Data**: YouTube Transcript API
- **Data Validation**: Pydantic
