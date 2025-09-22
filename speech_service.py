"""
Speech and Language Processing Service for INGRES ChatBot
Handles speech-to-text, text-to-text translation, and text-to-speech conversion
"""

import os
import base64
import tempfile
import logging
from typing import Dict, Optional, Tuple, Union
from io import BytesIO
from sarvamai import SarvamAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SpeechLanguageService:
    """Service for handling speech and language processing"""
    
    def __init__(self):
        """Initialize the speech and language service"""
        self.api_key = os.getenv("SARVAM_API_KEY", "")
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not found. Speech functionality will be disabled.")
            self.client = None
        else:
            try:
                self.client = SarvamAI(api_subscription_key=self.api_key)
                logger.info("Sarvam AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Sarvam AI client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if the speech service is available"""
        return self.client is not None
    
    def detect_language_from_text(self, text: str) -> Optional[str]:
        """Detect language from text input"""
        if not self.client:
            return None
        
        try:
            response = self.client.text.identify_language(input=text)
            language_code = getattr(response, 'language_code', None)
            logger.info(f"Detected language: {language_code} for text: {text[:50]}...")
            return language_code
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return None
    
    def speech_to_text(self, audio_file: Union[str, bytes, BytesIO]) -> Dict[str, Optional[str]]:
        """Convert speech to text and detect language"""
        if not self.client:
            return {"transcript": None, "language_code": None, "error": "Speech service not available"}
        
        try:
            # Handle different input types
            if isinstance(audio_file, str):
                # File path
                with open(audio_file, "rb") as f:
                    file_obj = f
                    response = self.client.speech_to_text.translate(
                        file=file_obj,
                        model="saaras:v2.5"
                    )
            elif isinstance(audio_file, bytes):
                # Bytes data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_file)
                    temp_file.flush()
                    
                    with open(temp_file.name, "rb") as f:
                        response = self.client.speech_to_text.translate(
                            file=f,
                            model="saaras:v2.5"
                        )
                    
                    os.unlink(temp_file.name)  # Clean up temp file
            elif isinstance(audio_file, BytesIO):
                # BytesIO object
                audio_file.seek(0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_file.read())
                    temp_file.flush()
                    
                    with open(temp_file.name, "rb") as f:
                        response = self.client.speech_to_text.translate(
                            file=f,
                            model="saaras:v2.5"
                        )
                    
                    os.unlink(temp_file.name)  # Clean up temp file
            else:
                return {"transcript": None, "language_code": None, "error": "Invalid audio file format"}
            
            # Extract transcript and language_code from response
            transcript = None
            language_code = None
            
            try:
                # If response is a dict-like object
                if isinstance(response, dict):
                    output = response.get("output") or response.get("data") or {}
                    
                    transcript = (
                        response.get("transcript") or response.get("text") or 
                        output.get("transcript") or output.get("text")
                    )
                    
                    language_code = (
                        response.get("language_code") or response.get("detected_language") or
                        output.get("language_code") or output.get("detected_language")
                    )
                else:
                    # Try attribute access for SDK objects
                    transcript = getattr(response, "transcript", None) or getattr(response, "text", None)
                    language_code = getattr(response, "language_code", None) or getattr(response, "detected_language", None)
            except Exception:
                pass
            
            logger.info(f"Speech to text conversion successful. Language: {language_code}")
            return {
                "transcript": transcript,
                "language_code": language_code,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in speech to text conversion: {e}")
            return {"transcript": None, "language_code": None, "error": str(e)}
    
    def translate_text(self, text: str, source_language: str, target_language: str = "en-IN") -> Dict[str, Optional[str]]:
        """Translate text from source language to target language"""
        if not self.client:
            return {"translated_text": None, "error": "Speech service not available"}
        
        try:
            response = self.client.text.translate(
                input=text,
                source_language_code=source_language,
                target_language_code=target_language,
                speaker_gender="Male"
            )
            
            # Extract translated text from response
            translated_text = None
            try:
                if isinstance(response, dict):
                    translated_text = response.get("text") or response.get("translated_text")
                else:
                    translated_text = getattr(response, "text", None) or getattr(response, "translated_text", None)
            except Exception:
                pass
            
            logger.info(f"Text translation successful: {source_language} -> {target_language}")
            return {
                "translated_text": translated_text or str(response),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            return {"translated_text": None, "error": str(e)}
    
    def text_to_speech(self, text: str, target_language: str, speaker: str = "anushka") -> Dict[str, Optional[Union[bytes, str]]]:
        """Convert text to speech in specified language"""
        if not self.client:
            return {"audio_data": None, "error": "Speech service not available"}
        
        try:
            response = self.client.text_to_speech.convert(
                text=text,
                model="bulbul:v2",
                target_language_code=target_language,
                speaker=speaker
            )
            
            # Extract audio data from response
            audio_data = None
            try:
                if hasattr(response, 'audios') and response.audios:
                    # Decode base64 audio data
                    audio_data = base64.b64decode("".join(response.audios))
                elif isinstance(response, dict) and response.get("audios"):
                    audio_data = base64.b64decode("".join(response["audios"]))
            except Exception as e:
                logger.error(f"Error extracting audio data: {e}")
            
            logger.info(f"Text to speech conversion successful for language: {target_language}")
            return {
                "audio_data": audio_data,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            return {"audio_data": None, "error": str(e)}
    
    def process_multilingual_chat_input(self, input_data: Dict) -> Dict:
        """
        Process chat input which can be text or voice
        Returns processed data with appropriate language handling
        """
        input_type = input_data.get("type", "text")  # "text" or "voice"
        
        if input_type == "text":
            text = input_data.get("text", "")
            if not text:
                return {"error": "No text provided"}
            
            # Detect language of input text
            detected_language = self.detect_language_from_text(text)
            
            return {
                "type": "text",
                "text": text,
                "language_code": detected_language or "en-IN",
                "error": None
            }
        
        elif input_type == "voice":
            audio_data = input_data.get("audio_data")
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Convert speech to text
            stt_result = self.speech_to_text(audio_data)
            if stt_result["error"]:
                return {"error": f"Speech to text failed: {stt_result['error']}"}
            
            return {
                "type": "voice",
                "text": stt_result["transcript"],
                "language_code": stt_result["language_code"] or "en-IN",
                "error": None
            }
        
        else:
            return {"error": "Invalid input type. Must be 'text' or 'voice'"}
    
    def process_multilingual_chat_output(self, response_text: str, target_language: str, 
                                       input_type: str, translate_to_english: bool = True) -> Dict:
        """
        Process chat output for multilingual response
        Returns appropriate response format based on input type and language
        """
        try:
            result = {
                "text_response": response_text,
                "translated_response": None,
                "audio_response": None,
                "target_language": target_language,
                "error": None
            }
            
            # If input was in a non-English language and we need translation
            if translate_to_english and target_language and target_language != "en-IN":
                # Translate response to target language
                translation_result = self.translate_text(
                    text=response_text,
                    source_language="en-IN",
                    target_language=target_language
                )
                
                if translation_result["error"]:
                    logger.warning(f"Translation failed: {translation_result['error']}")
                    result["translated_response"] = response_text  # Fallback to original
                else:
                    result["translated_response"] = translation_result["translated_text"]
            else:
                result["translated_response"] = response_text
            
            # If input was voice, generate audio response
            if input_type == "voice":
                text_for_speech = result["translated_response"] or response_text
                tts_result = self.text_to_speech(
                    text=text_for_speech,
                    target_language=target_language or "en-IN"
                )
                
                if tts_result["error"]:
                    logger.warning(f"Text to speech failed: {tts_result['error']}")
                    result["audio_response"] = None
                else:
                    result["audio_response"] = tts_result["audio_data"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing multilingual chat output: {e}")
            return {
                "text_response": response_text,
                "translated_response": None,
                "audio_response": None,
                "target_language": target_language,
                "error": str(e)
            }


# Language code mappings for common languages
LANGUAGE_MAPPINGS = {
    "hindi": "hi-IN",
    "english": "en-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "punjabi": "pa-IN"
}


def get_language_code(language_name: str) -> str:
    """Get language code from language name"""
    return LANGUAGE_MAPPINGS.get(language_name.lower(), "en-IN")