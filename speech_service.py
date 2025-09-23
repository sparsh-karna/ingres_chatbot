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
                logger.info(f"Processing audio file: {audio_file}")
                with open(audio_file, "rb") as f:
                    response = self.client.speech_to_text.translate(
                        file=f,
                        model="saaras:v1"
                    )
            elif isinstance(audio_file, (bytes, BytesIO)):
                # Bytes data or BytesIO object
                if isinstance(audio_file, BytesIO):
                    audio_file.seek(0)
                    audio_data = audio_file.read()
                else:
                    audio_data = audio_file
                
                logger.info(f"Processing audio bytes: {len(audio_data)} bytes")
                
                # Create a temporary file with proper extension
                # Use .webm extension since that's what the browser sends
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    logger.info(f"Created temporary file: {temp_file.name}")
                    
                    try:
                        with open(temp_file.name, "rb") as f:
                            logger.info("Sending audio to Sarvam AI for transcription...")
                            response = self.client.speech_to_text.translate(
                                file=f,
                                model="saaras:v1"
                            )
                            logger.info(f"Sarvam AI response received: {type(response)}")
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_file.name)
                            logger.info("Temporary file cleaned up")
                        except:
                            pass
            else:
                return {"transcript": None, "language_code": None, "error": "Invalid audio file format"}
            
            # Extract transcript and language_code from response
            transcript = None
            language_code = None
            
            logger.info(f"Processing Sarvam AI response: {response}")
            
            try:
                # Check if response has direct attributes first (SDK object)
                if hasattr(response, 'transcript'):
                    transcript = response.transcript
                elif hasattr(response, 'text'):
                    transcript = response.text
                elif hasattr(response, 'output'):
                    # Check if output has transcript
                    output = response.output
                    if hasattr(output, 'transcript'):
                        transcript = output.transcript
                    elif hasattr(output, 'text'):
                        transcript = output.text
                
                # Try to get language code
                if hasattr(response, 'language_code'):
                    language_code = response.language_code
                elif hasattr(response, 'detected_language'):
                    language_code = response.detected_language
                elif hasattr(response, 'output'):
                    output = response.output
                    if hasattr(output, 'language_code'):
                        language_code = output.language_code
                    elif hasattr(output, 'detected_language'):
                        language_code = output.detected_language
                
                # If SDK approach didn't work, try dict-like access
                if not transcript and isinstance(response, dict):
                    transcript = (
                        response.get("transcript") or response.get("text") or 
                        response.get("output", {}).get("transcript") or
                        response.get("output", {}).get("text")
                    )
                    
                if not language_code and isinstance(response, dict):
                    language_code = (
                        response.get("language_code") or response.get("detected_language") or
                        response.get("output", {}).get("language_code") or
                        response.get("output", {}).get("detected_language")
                    )
                    
            except Exception as extract_error:
                logger.error(f"Error extracting data from response: {extract_error}")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response content: {response}")
            
            if transcript:
                logger.info(f"Speech to text successful. Transcript: '{transcript}' Language: {language_code}")
                return {
                    "transcript": transcript,
                    "language_code": language_code or "en-IN",
                    "error": None
                }
            else:
                logger.error(f"No transcript found in response: {response}")
                return {"transcript": None, "language_code": None, "error": "No transcript found in Sarvam AI response"}
            
        except Exception as e:
            logger.error(f"Error in speech to text conversion: {e}")
            return {"transcript": None, "language_code": None, "error": str(e)}
    
    def translate_text(self, text: str, source_language: str, target_language: str = "en-IN") -> Dict[str, Optional[str]]:
        """Translate text from source language to target language for TTS only"""
        if not self.client:
            return {"translated_text": text, "error": "Translation service not available"}
        
        # If same language, no translation needed
        if source_language == target_language:
            return {"translated_text": text, "error": None}
            
        try:
            logger.info(f"Translating text for TTS: '{text[:50]}...' from {source_language} to {target_language}")
            
            # Split text into smaller chunks if it's too long (1000 char limit)
            max_chunk_size = 900  # Leave some buffer
            if len(text) <= max_chunk_size:
                return self._translate_single_text(text, source_language, target_language)
            else:
                # Split into sentences and group them into chunks
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Translate each chunk
                translated_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Translating chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                    result = self._translate_single_text(chunk, source_language, target_language)
                    if result["error"]:
                        logger.error(f"Chunk {i+1} translation failed: {result['error']}")
                        return {"translated_text": text, "error": f"Translation failed at chunk {i+1}"}
                    translated_chunks.append(result["translated_text"])
                
                final_translation = " ".join(translated_chunks)
                logger.info(f"Long text translation completed: {len(text)} -> {len(final_translation)} chars")
                return {"translated_text": final_translation, "error": None}
                
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            return {"translated_text": text, "error": str(e)}
    
    def _translate_single_text(self, text: str, source_language: str, target_language: str) -> Dict[str, Optional[str]]:
        """Translate a single text chunk"""
        try:
            logger.info(f"Translating text: '{text[:50]}...' from {source_language} to {target_language}")
            response = self.client.text.translate(
                input=text,
                source_language_code=source_language,
                target_language_code=target_language,
                model="mayura:v1",
                speaker_gender="Male"
            )
            
            # Extract translated text from response
            translated_text = getattr(response, 'translated_text', None)
            if not translated_text and isinstance(response, dict):
                translated_text = response.get('translated_text')
            
            logger.info(f"Translation API response: {response}")
            
            if translated_text:
                logger.info(f"Text translation successful: {source_language} -> {target_language}")
                logger.info(f"Translated text: '{translated_text[:100]}...'")
                return {"translated_text": translated_text, "error": None}
            else:
                logger.error(f"No translated text found in response: {response}")
                return {"translated_text": text, "error": "No translation found in response"}
                
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"translated_text": text, "error": str(e)}

    
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
        Process chat output for voice response
        Returns appropriate response format based on input type
        """
        try:
            result = {
                "text_response": response_text,
                "translated_response": None,  # Will be set if we need translation for TTS
                "audio_response": None,
                "target_language": target_language,
                "error": None
            }
            
            # If input was voice, generate audio response in the same language
            if input_type == "voice":
                logger.info(f"Generating audio response in language: {target_language}")
                
                # For non-English languages, we need to translate the response for TTS
                text_for_tts = response_text
                if target_language != "en-IN":
                    logger.info(f"Translating response from English to {target_language} for TTS")
                    translation_result = self.translate_text(
                        text=response_text,
                        source_language="en-IN", 
                        target_language=target_language
                    )
                    
                    if translation_result["error"]:
                        logger.warning(f"Translation failed, using English for TTS: {translation_result['error']}")
                        text_for_tts = response_text
                        result["translated_response"] = response_text
                    else:
                        text_for_tts = translation_result["translated_text"]
                        result["translated_response"] = translation_result["translated_text"]
                        logger.info(f"Translation successful for TTS")
                else:
                    result["translated_response"] = response_text
                
                # Generate audio in the target language
                tts_result = self.text_to_speech(
                    text=text_for_tts,
                    target_language=target_language or "en-IN"
                )
                
                if tts_result["error"]:
                    logger.warning(f"Text to speech failed: {tts_result['error']}")
                    result["audio_response"] = None
                else:
                    result["audio_response"] = tts_result["audio_data"]
                    logger.info(f"Audio response generated successfully in {target_language}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chat output: {e}")
            return {
                "text_response": response_text,
                "translated_response": response_text,
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