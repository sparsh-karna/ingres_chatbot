from sarvamai import SarvamAI
import os
import html
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")
client = SarvamAI(
    api_subscription_key=API_KEY
)

response_text = client.text.translate(
    input="Hello, how are you?",
    source_language_code="en-IN",
    target_language_code="mr-IN",
    speaker_gender="Male"
)

print(response_text)


# Extract the translated text string from Sarvam response
translated_text = (
    response_text if isinstance(response_text, str)
    else (
        getattr(response_text, "text", None)
        or getattr(response_text, "translated_text", None)
        or (response_text.get("text") if isinstance(response_text, dict) else None)
        or (response_text.get("translated_text") if isinstance(response_text, dict) else None)
    )
)

if not translated_text:
    raise RuntimeError("No translated text available for TTS")

if not AZURE_API_KEY:
    raise RuntimeError("AZURE_API_KEY not set in environment")

# Map language code to an Azure Neural voice
def get_azure_voice(language_code: str) -> str:
    mapping = {
        "en-IN": "en-IN-NeerjaNeural",
        "hi-IN": "hi-IN-SwaraNeural",
        "ta-IN": "ta-IN-PallaviNeural",
        "te-IN": "te-IN-ShrutiNeural",
        "bn-IN": "bn-IN-TanishaaNeural",
        "gu-IN": "gu-IN-DhwaniNeural",
        "kn-IN": "kn-IN-SapnaNeural",
        "ml-IN": "ml-IN-MidhunNeural",
        "mr-IN": "mr-IN-AarohiNeural",
        "pa-IN": "pa-IN-BaljeetNeural",
    }
    return mapping.get(language_code, "en-IN-NeerjaNeural")

voice_name = get_azure_voice("mr-IN")
endpoint = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

ssml = (
    f"<speak version='1.0' xml:lang='en-US'>"
    f"<voice name='{voice_name}'>"
    f"{html.escape(translated_text)}"
    f"</voice>"
    f"</speak>"
)

headers = {
    "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
    "Content-Type": "application/ssml+xml",
    "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
    "User-Agent": "ingres-tts-client"
}

resp = requests.post(endpoint, headers=headers, data=ssml.encode("utf-8"), timeout=30)
resp.raise_for_status()

with open("output.wav", "wb") as f:
    f.write(resp.content)