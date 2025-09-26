"""
INGRES AI ChatBot - FastAPI Backend (Modular Version)
RESTful API for querying groundwater resource data
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import base64
import requests
from pydantic import BaseModel

# Import modules
from database_manager import DatabaseManager
from query_processor import QueryProcessor
from speech_service import SpeechLanguageService
from models import (
    QueryRequest, ChatRequest, QueryResponse, ChatResponse,
    SessionResponse, SessionsListResponse, ChatHistoryResponse,
    HealthResponse, CSVData, CSVForecastDataInput, EDADataInput,
    GeneralChatRequest, GeneralChatResponse, LandAssessmentRequest, LandAssessmentResponse
)
from routes import Routes
from helpers import clean_md, land_assessment_analysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Microservices configuration
SERVICES = {
    "query_processor": "http://localhost:8001",
    "code_executor": "http://localhost:8002", 
    "result_analyzer": "http://localhost:8003"
}

# Default files for microservices
DEFAULT_CSV_FILE = "2024-2025.csv"
DEFAULT_INDEX_FILE = "index_2024-2025.txt"

# Pydantic models for microservices
class TestQuery(BaseModel):
    query: str = "Which state has highest rainfall?"

class ServiceStatus(BaseModel):
    service: str
    status: str
    endpoints: dict
    description: str
    usage: str

class MicroservicesHealthResponse(BaseModel):
    status: str
    services: dict
    csv_file: str
    index_file: str

# Global variables for database and processors
db_manager = None
query_processor = None
llm = None
speech_service = None
route_handlers = None

# Microservices helper functions
def check_service_health(service_url: str) -> bool:
    """Check if a microservice is running"""
    try:
        response = requests.get(f"{service_url}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_crisp_summary(analysis_result: dict) -> str:
    """Extract a crisp summary from analysis result"""
    try:
        # Try to get the most important information
        explanation = analysis_result.get('explanation', '')
        key_insights = analysis_result.get('key_insights', [])

        # Start with the first key insight if available
        if key_insights and len(key_insights) > 0:
            summary = key_insights[0]
        else:
            # Fallback to explanation
            summary = explanation

        # Clean and limit to 160 characters
        summary = summary.strip()
        if len(summary) > 160:
            summary = summary[:157] + "..."

        return summary

    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return "Analysis completed successfully."

def process_query_with_microservices(query: str) -> str:
    """Process query using microservices pipeline"""
    try:
        start_time = time.time()

        # Step 1: Generate code using query processor
        logger.info(f"Generating code for query: {query}")
        query_payload = {
            "query": query,
            "index_file": DEFAULT_INDEX_FILE,
            "csv_file": DEFAULT_CSV_FILE
        }

        code_response = requests.post(
            f"{SERVICES['query_processor']}/generate-code",
            json=query_payload,
            timeout=30
        )

        if code_response.status_code != 200:
            logger.error(f"Code generation failed: {code_response.status_code}")
            return "Sorry, I couldn't generate code for your query."

        code_result = code_response.json()
        generated_code = code_result.get('code', '')

        if not generated_code:
            return "Sorry, no code was generated for your query."

        # Step 2: Execute the code
        logger.info("Executing generated code")
        execution_payload = {
            "code": generated_code,
            "csv_file": DEFAULT_CSV_FILE,
            "include_analysis": True,
            "original_query": query
        }

        execution_response = requests.post(
            f"{SERVICES['code_executor']}/execute-with-analysis",
            json=execution_payload,
            timeout=60
        )

        if execution_response.status_code != 200:
            logger.error(f"Code execution failed: {execution_response.status_code}")
            return "Sorry, I couldn't execute the analysis."

        execution_result = execution_response.json()

        if not execution_result.get('success', False):
            error_msg = execution_result.get('error', 'Unknown error')
            logger.error(f"Execution failed: {error_msg}")
            return "Sorry, the analysis failed to complete."

        # Step 3: Get analysis if not included in execution
        analysis_result = execution_result.get('analysis')
        if not analysis_result:
            # Call result analyzer separately
            logger.info("Getting analysis from result analyzer")
            analysis_payload = {
                "query": query,
                "print_output": execution_result.get('print_output', ''),
                "dataframes": execution_result.get('dataframes', []),
                "variables": execution_result.get('variables', {}),
                "execution_time": execution_result.get('execution_time', 0)
            }

            analysis_response = requests.post(
                f"{SERVICES['result_analyzer']}/analyze-results",
                json=analysis_payload,
                timeout=30
            )

            if analysis_response.status_code == 200:
                analysis_result = analysis_response.json()

        # Step 4: Create crisp summary
        if analysis_result:
            summary = get_crisp_summary(analysis_result)
        else:
            # Fallback: extract from print output
            print_output = execution_result.get('print_output', '')
            if print_output:
                lines = print_output.strip().split('\n')
                summary = lines[-1] if lines else "Analysis completed"
                if len(summary) > 160:
                    summary = summary[:157] + "..."
            else:
                summary = "Analysis completed successfully"

        total_time = time.time() - start_time
        logger.info(f"Query processed in {total_time:.2f}s: {summary}")

        return summary

    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return "Sorry, the analysis is taking too long. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return "Sorry, I'm having trouble connecting to the analysis services."
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "Sorry, there was an error processing your request."

def get_microservices_response(query: str) -> str:
    """Get a response using microservices pipeline"""
    try:
        # Check if required services are running
        required_services = ['query_processor', 'code_executor']
        for service_name in required_services:
            if not check_service_health(SERVICES[service_name]):
                logger.error(f"Service {service_name} is not available")
                return f"Sorry, the {service_name.replace('_', ' ')} service is not available."

        # Process query using microservices
        response = process_query_with_microservices(query)
        return response

    except Exception as e:
        logger.error(f"Error getting microservices response: {e}")
        return "Sorry, there was an error processing your request."


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


@app.post("/general-chat", response_model=GeneralChatResponse)
async def process_general_chat(request: GeneralChatRequest):
    """Process general questions about the groundwater dataset using LLM"""
    return await route_handlers.process_general_chat(request)


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

@app.post("/forecast")
async def forecast(csv_data: CSVForecastDataInput):
    return await route_handlers.forecast(csv_data)

@app.post("/eda")
async def eda(eda_input: EDADataInput):
    return await route_handlers.eda(eda_input)

@app.post("/land-assessment", response_model=LandAssessmentResponse)
async def land_assessment(request: LandAssessmentRequest):
    """
    Analyze land and crop suitability based on location and farming parameters.
    
    This endpoint provides comprehensive analysis including:
    - Monthly water requirements 
    - Crop suitability scores
    - Soil analysis
    - Water sources distribution
    - Farming recommendations
    """
    try:
        # Call the land assessment analysis function
        analysis_result = land_assessment_analysis(
            state=request.state,
            district=request.district,
            assessment_unit=request.assessment_unit,
            cropping_season=request.cropping_season,
            soil_type=request.soil_type,
            irrigation_type=request.irrigation_type
        )
        
        # Create response object
        response = LandAssessmentResponse(
            success=True,
            water_requirements=analysis_result.get('water_requirements', []),
            crop_suitability=analysis_result.get('crop_suitability', []),
            soil_analysis=analysis_result.get('soil_analysis'),
            water_sources=analysis_result.get('water_sources'),
            recommendations=analysis_result.get('recommendations', []),
            total_annual_requirement=analysis_result.get('total_annual_requirement'),
            critical_months=analysis_result.get('critical_months', []),
            error=analysis_result.get('error', '')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in land assessment endpoint: {e}")
        
        # Return error response
        return LandAssessmentResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

# --- Microservices Integration Endpoints ---
@app.get("/microservices/status", response_model=ServiceStatus)
async def microservices_status():
    """Get microservices status information"""
    return ServiceStatus(
        service="INGRES ChatBot with Microservices Integration",
        status="running",
        endpoints={
            "/microservices/webhook": "POST - Microservices WhatsApp webhook",
            "/microservices/test": "GET/POST - Test the microservices system",
            "/microservices/health": "GET - Health check for all microservices"
        },
        description="Send WhatsApp messages to analyze groundwater and rainfall data using microservices",
        usage="Send a message to the WhatsApp number to get policy analysis via microservices"
    )

@app.post("/microservices/webhook")
async def microservices_webhook(From: str = Form(default=""), To: str = Form(default=""), Body: str = Form(default=""), MessageSid: str = Form(default="")):
    """Microservices WhatsApp webhook: uses microservices pipeline for processing"""
    try:
        # Log the incoming request for debugging
        logger.info(f"Microservices webhook received - From: {From}, To: {To}, Body: {Body[:100]}...")
        
        incoming_msg = Body.strip()
        
        if not incoming_msg:
            response_text = "Please send a question about groundwater or rainfall data."
        else:
            # Get response using microservices
            response_text = get_microservices_response(incoming_msg)

        # Clean and truncate response
        response_text = response_text.strip()
        if len(response_text) > 1600:
            response_text = response_text[:1597] + "..."

        # For WhatsApp, we need to specify the To field in the Message tag
        if From.startswith("whatsapp:"):
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message to="{From}">{_xml_escape(response_text)}</Message>
</Response>"""
        else:
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{_xml_escape(response_text)}</Message>
</Response>"""
        
        logger.info(f"Returning microservices TwiML response: {twiml[:200]}...")
        return Response(content=twiml, media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Error in microservices webhook: {e}")
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sorry, an error occurred: {_xml_escape(str(e))}</Message>
</Response>"""
        return Response(content=twiml, media_type="text/xml")

@app.get("/microservices/test")
async def microservices_test_get():
    """Test endpoint to verify microservices are running"""
    return {"status": "Microservices integration is running", "endpoint": "/microservices/webhook"}

@app.post("/microservices/test")
async def microservices_test_post(query_data: TestQuery):
    """Test endpoint with microservices query processing"""
    response = get_microservices_response(query_data.query)
    return {"query": query_data.query, "response": response}

@app.get("/microservices/health", response_model=MicroservicesHealthResponse)
async def microservices_health():
    """Health check endpoint for microservices"""
    # Check service health
    service_status = {}
    for service_name, service_url in SERVICES.items():
        service_status[service_name] = check_service_health(service_url)

    all_healthy = all(service_status.values())

    return MicroservicesHealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=service_status,
        csv_file=DEFAULT_CSV_FILE,
        index_file=DEFAULT_INDEX_FILE
    )


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
async def twilio_sms(From: str = Form(default=""), To: str = Form(default=""), Body: str = Form(default=""), MessageSid: str = Form(default="")):
    """Twilio SMS/WhatsApp webhook: returns plain text response only via TwiML."""
    try:
        from models import ChatRequest  # local import to avoid circulars at import time
        
        # Log the incoming request for debugging
        logger.info(f"Twilio webhook received - From: {From}, To: {To}, Body: {Body[:100]}...")
        
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

        # For WhatsApp, we need to specify the To field in the Message tag
        if From.startswith("whatsapp:"):
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message to="{From}">{_xml_escape(message_text)}</Message>
</Response>"""
        else:
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{_xml_escape(message_text)}</Message>
</Response>"""
        
        logger.info(f"Returning TwiML response: {twiml[:200]}...")
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        logger.error(f"Error in Twilio webhook: {e}")
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sorry, an error occurred: {_xml_escape(str(e))}</Message>
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# --- Twilio Voice Call Integration ---
@app.post("/twilio/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice call - initial greeting"""
    try:
        # Get form data
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        
        logger.info(f"Incoming voice call from {from_number} to {to_number}")
        
        # TwiML for initial greeting and recording
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">
        Hello! Welcome to INGRES Groundwater Analysis System. 
        Please speak your question about groundwater data after the beep.
    </Say>
    <Record 
        action="/twilio/voice/process" 
        method="POST"
        maxLength="30"
        finishOnKey="#"
        playBeep="true"
        recordingStatusCallback="/twilio/voice/recording-status"
    />
    <Say voice="alice" language="en-IN">
        I didn't hear anything. Please call back and try again.
    </Say>
</Response>"""
        
        return Response(content=twiml, media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">
        Sorry, there was an error processing your call. Please try again later.
    </Say>
</Response>"""
        return Response(content=twiml, media_type="text/xml")

@app.post("/twilio/voice/process")
async def process_voice_input(request: Request):
    """Process recorded voice input and generate speech response"""
    try:
        # Get form data
        form_data = await request.form()
        recording_url = form_data.get("RecordingUrl", "")
        from_number = form_data.get("From", "")
        
        logger.info(f"Processing voice input from {from_number}, recording URL: {recording_url}")
        
        if not recording_url:
            return _voice_error_response("No recording URL provided")
        
        if not speech_service or not speech_service.is_available():
            return _voice_error_response("Speech service not available")
        
        # Download and process the recording
        try:
            # Download the recording
            response = requests.get(recording_url, timeout=30)
            response.raise_for_status()
            audio_data = response.content
            
            logger.info(f"Downloaded recording: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"Error downloading recording: {e}")
            return _voice_error_response("Failed to download recording")
        
        # Convert speech to text
        try:
            stt_result = speech_service.speech_to_text(audio_data)
            
            if not stt_result.get("transcript"):
                return _voice_error_response("Could not understand your speech. Please try speaking more clearly.")
            
            transcript = stt_result["transcript"]
            detected_language = stt_result.get("language_code", "en-IN")
            
            logger.info(f"Speech-to-text result: '{transcript}' (Language: {detected_language})")
            
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return _voice_error_response("Error processing your speech")
        
        # Process the query
        try:
            # Translate to English for SQL processing if needed
            question_for_sql = transcript
            if detected_language != "en-IN" and speech_service.is_translation_available():
                translation_result = speech_service.translate_text(
                    text=transcript,
                    source_language=detected_language,
                    target_language="en-IN"
                )
                if translation_result.get("translated_text") and not translation_result.get("error"):
                    question_for_sql = translation_result["translated_text"]
                    logger.info(f"Translated for SQL processing: '{question_for_sql}'")
            
            # Process the query
            result = query_processor.process_user_query(
                question_for_sql,
                include_visualization=False
            )
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error occurred')
                logger.error(f"Query processing failed: {error_msg}")
                return _voice_error_response(f"Sorry, I couldn't process your question: {error_msg}")
            
            response_text = result.get('response', 'No response generated')
            logger.info(f"Query processed successfully: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return _voice_error_response("Error processing your question")
        
        # Generate speech response
        try:
            # Clean the response text
            clean_response = clean_md(response_text)
            
            # Generate audio response in the detected language
            tts_result = speech_service.text_to_speech(
                text=clean_response,
                target_language=detected_language
            )
            
            if not tts_result.get("audio_data"):
                # Fallback to text response if TTS fails
                logger.warning("TTS failed, using text response")
                return _voice_text_response(clean_response)
            
            # Convert audio to base64 for TwiML
            audio_base64 = base64.b64encode(tts_result["audio_data"]).decode('utf-8')
            
            # Create TwiML with audio response
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>data:audio/mp3;base64,{audio_base64}</Play>
    <Say voice="alice" language="{detected_language}">
        Thank you for using INGRES Groundwater Analysis System. Goodbye!
    </Say>
</Response>"""
            
            logger.info("Voice response generated successfully")
            return Response(content=twiml, media_type="text/xml")
            
        except Exception as e:
            logger.error(f"Error generating speech response: {e}")
            # Fallback to text response
            return _voice_text_response(clean_response)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        return _voice_error_response("An error occurred while processing your request")

@app.post("/twilio/voice/recording-status")
async def recording_status(request: Request):
    """Handle recording status callback"""
    try:
        form_data = await request.form()
        status = form_data.get("RecordingStatus", "")
        recording_url = form_data.get("RecordingUrl", "")
        
        logger.info(f"Recording status: {status}, URL: {recording_url}")
        
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error handling recording status: {e}")
        return Response(content="OK", media_type="text/plain")

def _voice_error_response(message: str) -> Response:
    """Generate error response TwiML for voice calls"""
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">
        {_xml_escape(message)}
    </Say>
    <Say voice="alice" language="en-IN">
        Thank you for calling. Goodbye!
    </Say>
</Response>"""
    return Response(content=twiml, media_type="text/xml")

def _voice_text_response(message: str) -> Response:
    """Generate text response TwiML for voice calls (fallback when TTS fails)"""
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">
        {_xml_escape(message)}
    </Say>
    <Say voice="alice" language="en-IN">
        Thank you for using INGRES Groundwater Analysis System. Goodbye!
    </Say>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_modular:app",
        host="0.0.0.0",
        port=8008,
        reload=True
    )