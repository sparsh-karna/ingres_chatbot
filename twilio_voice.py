"""
Twilio Voice Call Integration for INGRES ChatBot
Handles incoming calls, speech-to-text, query processing, and text-to-speech response
"""

import os
import logging
import requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from dotenv import load_dotenv
from speech_service import SpeechLanguageService
from query_processor import QueryProcessor
from database_manager import DatabaseManager
from helpers import clean_md
import asyncio
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Twilio Voice Integration")

# Global variables
speech_service = None
query_processor = None
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global speech_service, query_processor, db_manager
    
    logger.info("Initializing Twilio Voice services...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize query processor
        query_processor = QueryProcessor(db_manager)
        
        # Initialize speech service
        speech_service = SpeechLanguageService()
        if speech_service.is_available():
            logger.info("Speech service initialized successfully")
        else:
            logger.warning("Speech service not available - voice functionality will be disabled")
        
        logger.info("Twilio Voice services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def _xml_escape(text: str) -> str:
    """Escape XML special characters"""
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

@app.post("/twilio/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice call - initial greeting"""
    try:
        # Get form data
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        
        logger.info(f"Incoming call from {from_number} to {to_number}")
        
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
            return _error_response("No recording URL provided")
        
        if not speech_service or not speech_service.is_available():
            return _error_response("Speech service not available")
        
        # Download and process the recording
        try:
            # Download the recording with Twilio authentication
            twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            
            if not twilio_account_sid or not twilio_auth_token:
                logger.error("Twilio credentials not found in environment variables")
                return _error_response("Twilio credentials not configured")
            
            # Download with HTTP Basic Auth
            response = requests.get(
                recording_url, 
                auth=(twilio_account_sid, twilio_auth_token),
                timeout=30
            )
            response.raise_for_status()
            audio_data = response.content
            
            logger.info(f"Downloaded recording: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"Error downloading recording: {e}")
            return _error_response("Failed to download recording")
        
        # Convert speech to text
        try:
            stt_result = speech_service.speech_to_text(audio_data)
            
            if not stt_result.get("transcript"):
                return _error_response("Could not understand your speech. Please try speaking more clearly.")
            
            transcript = stt_result["transcript"]
            detected_language = stt_result.get("language_code", "en-IN")
            
            logger.info(f"Speech-to-text result: '{transcript}' (Language: {detected_language})")
            
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return _error_response("Error processing your speech")
        
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
                return _error_response(f"Sorry, I couldn't process your question: {error_msg}")
            
            response_text = result.get('response', 'No response generated')
            logger.info(f"Query processed successfully: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return _error_response("Error processing your question")
        
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
                return _text_response(clean_response)
            
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
            return _text_response(clean_response)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        return _error_response("An error occurred while processing your request")

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

def _error_response(message: str) -> Response:
    """Generate error response TwiML"""
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

def _text_response(message: str) -> Response:
    """Generate text response TwiML (fallback when TTS fails)"""
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

@app.get("/twilio/voice/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "speech_service": speech_service.is_available() if speech_service else False,
        "query_processor": query_processor is not None,
        "database": db_manager is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "twilio_voice:app",
        host="0.0.0.0",
        port=8002,  # Different port from main app
        reload=True
    )
