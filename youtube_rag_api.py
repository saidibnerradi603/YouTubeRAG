from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl,Field,field_validator
import os
import logging
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_google_genai import GoogleGenerativeAI
from pinecone import (
    Pinecone,
    ServerlessSpec,

)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize API keys from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
INDEX_NAME = "youtube-rag"

if not all([PINECONE_API_KEY, COHERE_API_KEY, GOOGLE_API_KEY]):
    raise EnvironmentError("Required API keys are missing. Please check your .env file.")


pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# Initialize embeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# Set up Q&A prompt template

prompt_template = """
You are an expert assistant.

Your task is to answer user questions using only the context provided below and what you know about the context provided  which comes from YouTube video content.

---

## RULES

1. Use ONLY the given **context**. Do NOT use external knowledge.
2. If the context is NOT relevant to the question, reply clearly:
   > "The provided context does not contain enough relevant information to answer this question."
3. Write the answer in detailed Markdown format.
4. Be precise, accurate, and well-structured.
5. DO NOT generate anything except the Markdown-formatted answer.

---

## VARIABLES

- {context}: Extracted information from a YouTube video.
- {question}: A user-submitted query.

---

## INSTRUCTIONS

(a) Read context carefully.  
(b) Think step-by-step about how each passage relates to question.  
(c) Draft a clear, thorough answer in Markdown that follows all RULES.  
(d) Output **only** the Markdown answer described.

Begin! (Do NOT output anything other than the Markdown answer.)
"""


prompt = PromptTemplate.from_template(prompt_template)


ytt_api = YouTubeTranscriptApi()


class VideoRequest(BaseModel):
    video_url: HttpUrl = Field(
        ...,
        description="Public YouTube video URL to process and embed in the vector database.",
        example="https://www.youtube.com/watch?v=aircAruvnKk"
    )
    language: str = Field(
        default="en",
        description="Transcript language code (e.g., 'en' for English, 'es' for Spanish). Used to fetch the correct transcript from YouTube.",
        example="en"
    )
    
    @field_validator("language")
    @classmethod  
    def validate_language(cls, v):
        allowed_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "ar"]
        if v not in allowed_languages:
            raise ValueError(f"Language must be one of: {', '.join(allowed_languages)}")
        return v

class ChatRequest(BaseModel):
    video_id: str = Field(
        ...,
        description="YouTube video ID (e.g., 'aircAruvnKk') that was processed via /process_video.",
        example="aircAruvnKk"
    )
    question: str = Field(
        ...,
        description="A natural language question related to the video's transcript content.",
        example="What are the main points discussed in the video?"
    )

class VideoResponse(BaseModel):
    video_id: str = Field(
        ...,
        description="The extracted YouTube video ID from the provided URL.",
        example="aircAruvnKk"
    )
    status: str = Field(
        ...,
        description="The result of the processing operation. Can be 'completed', 'failed'.",
        example="completed"
    )
    message: str = Field(
        ...,
        description="Detailed message about the result, such as whether the video was already processed or if an error occurred.",
        example="Video has been successfully processed and is ready for querying."
    )

class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The generated answer based on the video content and the user's question.",
        example="The video explains how neural networks function using layered perceptrons."
    )


# Initialize FastAPI
app = FastAPI(
    title="YouTube RAG API",
    description="API for question answering on YouTube videos",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-youtube-app.streamlit.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Helper function to extract video ID from YouTube URL
def extract_video_id(video_url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    parsed_url = urlparse(video_url)
    
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [''])[0]
    
    elif parsed_url.netloc == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    return ""

# Helper function  to check if the video exists in Pinecone 
def check_video_exists(video_id: str) -> bool:
    try:
        zero_vector = [0.0] * 1024
        resp = index.query(
            top_k=1,
            vector=zero_vector,
            filter={"source": {"$eq": video_id}},
            include_metadata=False
        )
        return len(resp.matches) > 0
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return False


# Process a YouTube video (Load -> Split -> Embeddings ->  Vector database)
def process_youtube_video(video_id: str, language: str = "en") -> bool:
    """
    Process a YouTube video:
    1. Load the transcript
    2. Split into chunks
    3. Create embeddings
    4. Store in vector database
    """
    try:
        logger.info(f"Starting processing for video {video_id}")
        
        # Step 1: Load the YouTube transcript using YouTube Transcript API
        try:
            # Get transcript using youtube-transcript-api
            transcript = ytt_api.fetch(video_id, languages=[language])
          
            # Combine all transcript segments into a single text
            full_transcript = ""
            for snippet in transcript.snippets:
                full_transcript += snippet.text + " "
            # Create Document object to match LangChain format
            documents = [Document(
                page_content=full_transcript,
                metadata={
                    'source': video_id,
                }
            )]
            
        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return False
        

        
        # Step 2: Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        
        logger.info(f"Split transcript into {len(docs)} chunks")
        
        #  Step 3 & 4: Create embeddings and store in vector database

        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        vector_store.add_documents(docs)

        
        logger.info(f"Successfully processed video {video_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return False

# Get an answer to a question about a video
def get_answer(video_id: str, question: str) -> str:
    """
    Get an answer to a question about a video:
    1. Retrieve relevant chunks from vector store
    2. Format context
    3. Generate answer using LLM
    """
    try:
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        # Create retriever specific to this video
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":7, "filter": {"source": video_id}}
            ),
        llm=llm
    
        )
        # Define function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        
        answer = rag_chain.invoke(question)
        return answer 
    
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        return f"Error: {str(e)}"

# API Endpoints
@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "ok", "message": "YouTube RAG API is running"}

@app.post("/process_video", response_model=VideoResponse)
def process_video_endpoint(request: VideoRequest):
    """
    Process a YouTube video by its URL.
    This will extract the transcript, chunk it, create embeddings,
    and store them in the vector database.
    """
    try:
        # Extract video ID from URL
        video_url_str = str(request.video_url)
        video_id = extract_video_id(video_url_str)
        
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Check if this video has already been processed
        if check_video_exists(video_id):
            return VideoResponse(
                video_id=video_id,
                status="completed",
                message="This video has already been processed and is ready for querying."
            )
        
        # Process video
        success = process_youtube_video(video_id, request.language)
        
        if success:
            return VideoResponse(
                video_id=video_id,
                status="completed",
                message="Video has been successfully processed and is ready for querying."
            )
        else:
            return VideoResponse(
                video_id=video_id,
                status="failed",
                message="Failed to process the video. Please check if the video has a transcript in the specified language."
            )
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Answer a question about a processed YouTube video.
    """
    try:
        video_id = request.video_id
        question = request.question
        
        # Check if the video exists in Pinecone
        if not check_video_exists(video_id):
            raise HTTPException(
                status_code=404,
                detail="Video not found. Please process this video first using the /process_video endpoint."
            )
        
        
        answer = get_answer(video_id, question)
        
        
        
        return ChatResponse(
            answer=answer
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

