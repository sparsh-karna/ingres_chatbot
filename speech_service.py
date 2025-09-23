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
        logger.info(f"Loading SARVAM_API_KEY: {'Found' if self.api_key else 'Not found'}")
        
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not found. Speech functionality will be disabled.")
            self.client = None
        else:
            try:
                logger.info(f"Attempting to initialize Sarvam AI client with API key: {self.api_key[:10]}...")
                self.client = SarvamAI(api_subscription_key=self.api_key)
                logger.info("Sarvam AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Sarvam AI client: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if the speech service is available"""
        return self.client is not None
    
    def detect_language_from_text(self, text: str) -> Optional[str]:
        """Detect language from text input"""
        if not self.client:
            # Fallback: simple language detection based on script
            return self._simple_language_detection(text)
        
        try:
            response = self.client.text.identify_language(input=text)
            language_code = getattr(response, 'language_code', None)
            logger.info(f"Detected language: {language_code} for text: {text[:50]}...")
            return language_code
        except Exception as e:
            logger.error(f"Error detecting language with Sarvam AI: {e}")
            # Fallback to simple detection
            return self._simple_language_detection(text)
    
    def _simple_language_detection(self, text: str) -> Optional[str]:
        """Simple language detection based on character patterns"""
        try:
            # Check for Devanagari script (Hindi)
            if any('\u0900' <= char <= '\u097F' for char in text):
                logger.info(f"Detected Hindi script in text: {text[:50]}...")
                return "hi-IN"
            
            # Check for Gujarati script
            if any('\u0A80' <= char <= '\u0AFF' for char in text):
                logger.info(f"Detected Gujarati script in text: {text[:50]}...")
                return "gu-IN"
            
            # Check for Bengali script
            if any('\u0980' <= char <= '\u09FF' for char in text):
                logger.info(f"Detected Bengali script in text: {text[:50]}...")
                return "bn-IN"
            
            # Check for Tamil script
            if any('\u0B80' <= char <= '\u0BFF' for char in text):
                logger.info(f"Detected Tamil script in text: {text[:50]}...")
                return "ta-IN"
            
            # Check for Telugu script
            if any('\u0C00' <= char <= '\u0C7F' for char in text):
                logger.info(f"Detected Telugu script in text: {text[:50]}...")
                return "te-IN"
            
            # Check for Kannada script
            if any('\u0C80' <= char <= '\u0CFF' for char in text):
                logger.info(f"Detected Kannada script in text: {text[:50]}...")
                return "kn-IN"
            
            # Check for Malayalam script
            if any('\u0D00' <= char <= '\u0D7F' for char in text):
                logger.info(f"Detected Malayalam script in text: {text[:50]}...")
                return "ml-IN"
            
            # Check for Punjabi script
            if any('\u0A00' <= char <= '\u0A7F' for char in text):
                logger.info(f"Detected Punjabi script in text: {text[:50]}...")
                return "pa-IN"
            
            # Default to English if no Indian script detected
            logger.info(f"No Indian script detected, defaulting to English for text: {text[:50]}...")
            return "en-IN"
            
        except Exception as e:
            logger.error(f"Error in simple language detection: {e}")
            return "en-IN"
    
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
            logger.warning(f"Sarvam AI client not available. Cannot translate from {source_language} to {target_language}")
            # Return the original text if translation service is not available
            return {"translated_text": text, "error": "Translation service not available - returning original text"}
        
        try:
            logger.info(f"Translating text: '{text[:50]}...' from {source_language} to {target_language}")
            response = self.client.text.translate(
                input=text,
                source_language_code=source_language,
                target_language_code=target_language,
                speaker_gender="Male"
            )
            
            logger.info(f"Translation API response: {response}")
            
            # Extract translated text from response
            translated_text = None
            try:
                if isinstance(response, dict):
                    # Try multiple possible keys
                    translated_text = (
                        response.get("text") or 
                        response.get("translated_text") or 
                        response.get("output", {}).get("text") or
                        response.get("output", {}).get("translated_text") or
                        response.get("data", {}).get("text") or
                        response.get("data", {}).get("translated_text")
                    )
                else:
                    # Try attribute access for SDK objects
                    translated_text = (
                        getattr(response, "text", None) or 
                        getattr(response, "translated_text", None) or
                        getattr(response, "output", None)
                    )
                    
                    # If output is an object, try to get text from it
                    if translated_text and hasattr(translated_text, "text"):
                        translated_text = getattr(translated_text, "text", None)
                    elif translated_text and isinstance(translated_text, dict):
                        translated_text = translated_text.get("text") or translated_text.get("translated_text")
                        
            except Exception as e:
                logger.error(f"Error extracting translated text: {e}")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response content: {response}")
            
            if not translated_text:
                logger.error(f"Could not extract translated text from response: {response}")
                return {"translated_text": None, "error": f"Could not extract translated text from API response"}
            
            logger.info(f"Text translation successful: {source_language} -> {target_language}")
            logger.info(f"Translated text: '{translated_text[:100]}...'")
            return {
                "translated_text": translated_text,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            # If we have a target language different from English, translate the response
            if target_language and target_language != "en-IN":
                logger.info(f"Translating response from en-IN to {target_language}")
                # Translate response to target language
                translation_result = self.translate_text(
                    text=response_text,
                    source_language="en-IN",
                    target_language=target_language
                )
                
                logger.info(f"Translation result: {translation_result}")
                
                if translation_result["error"]:
                    logger.warning(f"Translation failed: {translation_result['error']}")
                    result["translated_response"] = response_text  # Fallback to original
                else:
                    result["translated_response"] = translation_result["translated_text"]
                    logger.info(f"Successfully translated to: {result['translated_response'][:100]}...")
            else:
                logger.info(f"No translation needed: target_language={target_language}")
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