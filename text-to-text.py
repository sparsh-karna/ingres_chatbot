from sarvamai import SarvamAI
import os
import requests
import html
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY", "")
GOOGLE_TRANSLATION_API = os.getenv("GOOGLE_TRANSLATION_API", "")

client = SarvamAI(
    api_subscription_key=API_KEY
)

input_text = "हम बहुत सुंदर है"
response = client.text.identify_language(
    input=input_text
)

print(f"Request ID: {response.request_id}")
print(f"Language Code: {response.language_code}")
print(f"Script Code: {response.script_code}")

if not GOOGLE_TRANSLATION_API:
    raise RuntimeError("GOOGLE_TRANSLATION_API not set in environment")

# Use Google Cloud Translation API (v2) for text-to-text
source_lang = response.language_code  # e.g., hi-IN
target_lang = "en-IN"

source_short = source_lang.split("-")[0] if source_lang else None
target_short = target_lang.split("-")[0] if target_lang else None

url = "https://translation.googleapis.com/language/translate/v2"
params = {"key": GOOGLE_TRANSLATION_API}
data = {
    "q": input_text,
    "format": "text",
}
if source_short:
    data["source"] = source_short
if target_short:
    data["target"] = target_short

resp = requests.post(url, params=params, data=data, timeout=20)
resp.raise_for_status()
payload = resp.json()
translations = payload.get("data", {}).get("translations", []) if isinstance(payload, dict) else []
translated_text = translations[0].get("translatedText") if translations else None

if not translated_text:
    raise RuntimeError(f"No translation returned by Google: {payload}")

print(html.unescape(translated_text))
