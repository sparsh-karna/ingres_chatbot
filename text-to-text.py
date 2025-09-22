from sarvamai import SarvamAI
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY", "")

client = SarvamAI(
    api_subscription_key=API_KEY
)

input_text = "हम बहुत सुंदर है"
response = client.text.identify_language(
    input=input_text
)

print(f"Request ID: {response.request_id}")
print(f"Language Code: {response.language_code}")  # Output: en-IN
print(f"Script Code: {response.script_code}")      # Output: Latn



response_text = client.text.translate(
    input=input_text,
    source_language_code=response.language_code,
    target_language_code="en-IN",
    speaker_gender="Male"
)

print(response_text)
