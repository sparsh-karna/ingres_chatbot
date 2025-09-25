"""
INGRES AI ChatBot - FastAPI Backend (Modular Version)
RESTful API for querying groundwater resource data
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Import modules
from database_manager import DatabaseManager
from query_processor import QueryProcessor
from speech_service import SpeechLanguageService
from models import (
    QueryRequest, ChatRequest, QueryResponse, ChatResponse,
    SessionResponse, SessionsListResponse, ChatHistoryResponse,
    HealthResponse, CSVData
)
from routes import Routes
from helpers import clean_md

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database and processors
db_manager = None
query_processor = None
llm = None
speech_service = None
route_handlers = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global db_manager, query_processor, llm, speech_service, route_handlers
    
    logger.info("Initializing INGRES ChatBot components...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize query processor
        query_processor = QueryProcessor(db_manager)
        
        # Initialize LLM for explanations
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            llm = ChatOpenAI(
                model="gpt-4o-mini-2024-07-18",
                openai_api_key=api_key,
                temperature=0.1
            )
        else:
            logger.warning("OPENAI_API_KEY not found, LLM explanations will be disabled")
            llm = None
        
        # Initialize speech service
        speech_service = SpeechLanguageService()
        if speech_service.is_available():
            logger.info("Speech service initialized successfully")
        else:
            logger.warning("Speech service not available - voice functionality will be disabled")
        
        # Initialize route handlers
        route_handlers = Routes(db_manager, query_processor, llm, speech_service)
        
        logger.info("INGRES ChatBot initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize INGRES ChatBot: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down INGRES ChatBot...")
    if db_manager:
        try:
            db_manager.close_connection()
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Create FastAPI application
app = FastAPI(
    title="INGRES ChatBot API",
    description="AI-powered groundwater resource analysis chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Route definitions
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return await route_handlers.root()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await route_handlers.health_check()


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return SQL results with optional visualization"""
    return await route_handlers.process_query(request)


@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Process chat message with context awareness"""
    return await route_handlers.process_chat(request)


@app.post("/chat/new-session", response_model=SessionResponse)
async def create_new_chat_session():
    """Create a new chat session"""
    return await route_handlers.create_new_chat_session()


@app.get("/chat/sessions", response_model=SessionsListResponse)
async def get_chat_sessions(limit: int = 50):
    """Get list of all chat sessions"""
    return await route_handlers.get_chat_sessions(limit)


@app.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, limit: int = 50):
    """Get chat history for a specific session"""
    return await route_handlers.get_chat_history(session_id, limit)


@app.get("/sql", response_model=QueryResponse)
async def execute_sql_directly(sql_query: str):
    """Execute SQL query directly (for advanced users)"""
    return await route_handlers.execute_sql_directly(sql_query)


@app.get("/frontend", response_class=FileResponse)
async def serve_frontend():
    """Serve the simple frontend HTML file"""
    return FileResponse("frontend_simple.html")

@app.post("/decide")
async def decide(csv_data: CSVData):
    return await route_handlers.decide(csv_data)


@app.get("/voice-frontend", response_class=FileResponse)
async def serve_voice_frontend():
    """Serve the voice-enabled frontend HTML file"""
    return FileResponse("frontend_voice.html")


# --- Twilio SMS webhook ---
def _xml_escape(text: str) -> str:
    try:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
        )
    except Exception:
        return text


@app.post("/twilio/sms")
async def twilio_sms(From: str = Form(default=""), To: str = Form(default=""), Body: str = Form(default="")):
    """Twilio SMS webhook: returns plain text response only via TwiML."""
    try:
        from models import ChatRequest  # local import to avoid circulars at import time
        chat_req = ChatRequest(
            question=Body or "",
            input_type="text",
            include_visualization=False,
        )
        chat_res = await route_handlers.process_chat(chat_req)

        if chat_res.success and chat_res.response:
            message_text = clean_md(chat_res.response)
        else:
            message_text = chat_res.error or "Sorry, I couldn't process that request."

        message_text = message_text.strip()
        if len(message_text) > 1600:
            message_text = message_text[:1597] + "..."

        twiml = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Message>{_xml_escape(message_text)}</Message></Response>"
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        twiml = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Message>Sorry, an error occurred: {_xml_escape(str(e))}</Message></Response>"
        return Response(content=twiml, media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_modular:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )