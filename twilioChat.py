from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
import requests
import os
from dotenv import load_dotenv
import logging
import time
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Twilio WhatsApp Bot for Policy Analysis",
    description="Send WhatsApp messages to analyze groundwater and rainfall data",
    version="1.0.0"
)

# Pydantic models
class TestQuery(BaseModel):
    query: str = "Which state has highest rainfall?"

class ServiceStatus(BaseModel):
    service: str
    status: str
    endpoints: dict
    description: str
    usage: str

class HealthResponse(BaseModel):
    status: str
    services: dict
    csv_file: str
    index_file: str

# Service endpoints
SERVICES = {
    "query_processor": "http://localhost:8001",
    "code_executor": "http://localhost:8002",
    "result_analyzer": "http://localhost:8003"
}

# Default files
DEFAULT_CSV_FILE = "2024-2025.csv"
DEFAULT_INDEX_FILE = "index_2024-2025.txt"

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

        # Clean and limit to 160 characters
        summary = summary.strip()
        if len(summary) > 160:
            summary = summary[:157] + "..."

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

def get_crisp_response(query: str) -> str:
    """Get a crisp response for WhatsApp using microservices"""
    try:
        # Check if required services are running
        required_services = ['query_processor', 'code_executor']
        for service_name in required_services:
            if not check_service_health(SERVICES[service_name]):
                logger.error(f"Service {service_name} is not available")
                return f"Sorry, the {service_name.replace('_', ' ')} service is not available."

        # Process query using microservices
        response = process_query_with_services(query)
        return response

    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return "Sorry, there was an error processing your request."

@app.get("/", response_model=ServiceStatus)
def home():
    """Home page with service information"""
    return ServiceStatus(
        service="Twilio WhatsApp Bot for Policy Analysis",
        status="running",
        endpoints={
            "/webhook": "POST - Twilio WhatsApp webhook",
            "/test": "GET/POST - Test the analysis system",
            "/health": "GET - Health check for all services"
        },
        description="Send WhatsApp messages to analyze groundwater and rainfall data",
        usage="Send a message to the WhatsApp number to get policy analysis"
    )

@app.post("/webhook")
def webhook(Body: str = Form(""), From: str = Form("")):
    """Handle incoming WhatsApp messages from Twilio"""
    try:
        # Get the message from Twilio
        incoming_msg = Body.strip()
        from_number = From

        logger.info(f"Received message from {from_number}: {incoming_msg}")

        # Create response
        resp = MessagingResponse()
        msg = resp.message()

        if not incoming_msg:
            msg.body("Please send a question about groundwater or rainfall data.")
            response_text = "Please send a question about groundwater or rainfall data."
        else:
            # Get crisp response
            response_text = get_crisp_response(incoming_msg)
            msg.body(response_text)

        logger.info(f"Sending response: {response_text}")
        return Response(content=str(resp), media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        resp = MessagingResponse()
        msg = resp.message()
        msg.body("Sorry, there was an error processing your request.")
        return Response(content=str(resp), media_type="application/xml")

@app.get("/test")
def test_get():
    """Test endpoint to verify the service is running"""
    return {"status": "Twilio WhatsApp Bot is running", "endpoint": "/webhook"}

@app.post("/test")
def test_post(query_data: TestQuery):
    """Test endpoint with query processing"""
    response = get_crisp_response(query_data.query)
    return {"query": query_data.query, "response": response}

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    # Check service health
    service_status = {}
    for service_name, service_url in SERVICES.items():
        service_status[service_name] = check_service_health(service_url)

    all_healthy = all(service_status.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=service_status,
        csv_file=DEFAULT_CSV_FILE,
        index_file=DEFAULT_INDEX_FILE
    )

if __name__ == "__main__":
    import uvicorn
    # For development - use ngrok for public URL in production
    port = int(os.getenv('PORT', 5001))  # Use 5001 as default, allow override
    uvicorn.run("twilioChat:app", host="0.0.0.0", port=port, reload=True)
