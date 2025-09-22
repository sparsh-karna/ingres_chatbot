from sarvamai import SarvamAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY", "")

client = SarvamAI(
    api_subscription_key=API_KEY,
)

response = client.speech_to_text.translate(
    file=open("Recording.wav", "rb"),
    model="saaras:v2.5"
)

# Extract transcript and language_code from response
transcript = None
language_code = None

try:
    # If response is a dict-like object
    if isinstance(response, dict):
        # Common locations
        output = response.get("output") or response.get("data") or {}

        # Transcript candidates
        transcript = (
            response.get("transcript")
            or response.get("text")
            or output.get("transcript")
            or output.get("text")
        )

        # Language code candidates
        language_code = (
            response.get("language_code")
            or response.get("detected_language")
            or output.get("language_code")
            or output.get("detected_language")
        )

    else:
        # Try attribute access for SDK objects
        transcript = getattr(response, "transcript", None) or getattr(response, "text", None)
        language_code = getattr(response, "language_code", None) or getattr(response, "detected_language", None)
except Exception:
    pass

print({
    "transcript": transcript,
    "language_code": language_code,
})