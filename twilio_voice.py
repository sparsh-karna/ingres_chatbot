"""
Twilio Voice Call Integration for INGRES ChatBot
Handles incoming calls, speech-to-text, query processing, and text-to-speech response
"""

import os
import logging
import requests
import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from dotenv import load_dotenv
from speech_service import SpeechLanguageService
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

# Service endpoints
SERVICES = {
    "query_processor": "http://localhost:8001",
    "code_executor": "http://localhost:8002",
    "result_analyzer": "http://localhost:8003"
}

# Default files
DEFAULT_CSV_FILE = "2024-2025.csv"
DEFAULT_INDEX_FILE = "index_2024-2025.txt"

# Global variables
speech_service = None

def check_service_health(service_url: str) -> bool:
    """Check if a service is running"""
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

        # Clean and limit to reasonable length for voice
        summary = summary.strip()
        if len(summary) > 500:
            summary = summary[:497] + "..."

        return summary

    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return "Analysis completed successfully."

def process_query_with_services(query: str) -> str:
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

        # Step 4: Create summary for voice response
        if analysis_result:
            summary = get_crisp_summary(analysis_result)
        else:
            # Fallback: extract from print output
            print_output = execution_result.get('print_output', '')
            if print_output:
                lines = print_output.strip().split('\n')
                summary = lines[-1] if lines else "Analysis completed"
                if len(summary) > 500:
                    summary = summary[:497] + "..."
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

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global speech_service
    
    logger.info("Initializing Twilio Voice services...")
    
    try:
        # Initialize speech service
        speech_service = SpeechLanguageService()
        if speech_service.is_available():
            logger.info("Speech service initialized successfully")
        else:
            logger.warning("Speech service not available - voice functionality will be disabled")
        
        # Check if required microservices are running
        required_services = ['query_processor', 'code_executor']
        for service_name in required_services:
            if check_service_health(SERVICES[service_name]):
                logger.info(f"✅ {service_name} service is available")
            else:
                logger.warning(f"⚠️ {service_name} service is not available")
        
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
            
            # Check if required services are running
            required_services = ['query_processor', 'code_executor']
            for service_name in required_services:
                if not check_service_health(SERVICES[service_name]):
                    logger.error(f"Service {service_name} is not available")
                    return _error_response(f"Sorry, the {service_name.replace('_', ' ')} service is not available.")
            
            # Process the query using microservices pipeline
            response_text = process_query_with_services(question_for_sql)
            
            if not response_text or response_text.startswith("Sorry,"):
                logger.error(f"Query processing failed: {response_text}")
                return _error_response("Sorry, I couldn't process your question. Please try again.")
            
            logger.info(f"Query processed successfully: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return _error_response("Error processing your question")
        
        # Generate speech response using Twilio's built-in TTS
        try:
            # Clean the response text
            clean_response = clean_md(response_text)
            
            # Use Twilio's built-in text-to-speech instead of custom audio
            # This is more reliable and doesn't require serving audio files
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="{detected_language}">
        {_xml_escape(clean_response)}
    </Say>
    <Say voice="alice" language="{detected_language}">
        Thank you for using INGRES Groundwater Analysis System. Goodbye!
    </Say>
</Response>"""
            
            logger.info("Voice response generated successfully using Twilio TTS")
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
    # Check service health
    service_status = {}
    for service_name, service_url in SERVICES.items():
        service_status[service_name] = check_service_health(service_url)

    all_healthy = all(service_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "speech_service": speech_service.is_available() if speech_service else False,
        "services": service_status,
        "csv_file": DEFAULT_CSV_FILE,
        "index_file": DEFAULT_INDEX_FILE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "twilio_voice:app",
        host="0.0.0.0",
        port=8010,  # Different port from main app
        reload=True
    )
