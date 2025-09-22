from sarvamai import SarvamAI
import base64
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY", "")
client = SarvamAI(
    api_subscription_key=API_KEY
)

response_text = client.text.translate(
    input="Hello, how are you?",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    speaker_gender="Male"
)

print(response_text)


client = SarvamAI(api_subscription_key=API_KEY)
response_speech = client.text_to_speech.convert(
    text=(
        response_text if isinstance(response_text, str)
        else (
            getattr(response_text, "text", None)
            or getattr(response_text, "translated_text", None)
            or (response_text.get("text") if isinstance(response_text, dict) else None)
            or (response_text.get("translated_text") if isinstance(response_text, dict) else None)
        )
    ),
    model="bulbul:v2",
    target_language_code="hi-IN",
    speaker="anushka"
)
with open("output.wav", "wb") as f:
    f.write(base64.b64decode("".join(response_speech.audios)))