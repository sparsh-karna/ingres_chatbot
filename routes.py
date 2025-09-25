"""
Route Handlers for INGRES ChatBot FastAPI Backend
"""

import json
import logging
import re
import time
import pandas as pd
from fastapi import HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime

from models import (
    CSVData, QueryRequest, ChatRequest, QueryResponse, ChatResponse, 
    SessionResponse, SessionsListResponse, ChatHistoryResponse, 
    HealthResponse, ErrorResponse
)
from helpers import (
    convert_numpy_types, get_or_create_session_id, add_message_to_session,
    get_structured_chat_context, create_context_aware_question,
    generate_enhanced_contextual_explanation, add_enhanced_message_to_session,
    get_enhanced_chat_context, prepare_response_data, create_response_metadata,
    format_csv_data, get_chat_history_for_response, validate_session_id,
    create_error_response, decide_graph_from_string, clean_md, forecast_data
)

logger = logging.getLogger(__name__)


class Routes:
    """Route handlers for the FastAPI application"""
    
    def __init__(self, db_manager, query_processor, llm, speech_service=None):
        self.db_manager = db_manager
        self.query_processor = query_processor
        self.llm = llm
        self.speech_service = speech_service
    
    async def root(self) -> Dict[str, str]:
        """Root endpoint"""
        return {
            "message": "INGRES ChatBot API",
            "version": "1.0.0",
            "documentation": "/docs",
            "endpoints": {
                "query": "/query",
                "chat": "/chat", 
                "new_session": "/chat/new-session",
                "sessions": "/chat/sessions",
                "session_history": "/chat/{session_id}/history",
                "health": "/health"
            }
        }
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        # Check PostgreSQL connection
        try:
            test_query = "SELECT 1"
            test_result = self.db_manager.execute_query(test_query)
            postgres_status = "connected" if not test_result.empty else "disconnected"
        except:
            postgres_status = "disconnected"
        
        # Check MongoDB connection
        mongodb_status = "connected" if self.db_manager.is_mongodb_available() else "disconnected"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            database_status=postgres_status,
            mongodb_status=mongodb_status
        )
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process SQL query endpoint"""
        try:
            logger.info(f"Processing query: {request.question}")
            
            # Process the query
            result = self.query_processor.process_user_query(
                request.question, 
                include_visualization=request.include_visualization
            )
            
            if not result.get('success', False):
                raise Exception(result.get('error', 'Unknown error occurred'))
            
            result_data = result.get('data', pd.DataFrame())
            visualization = result.get('visualization', None)
            base_response = result.get('response', '')
            sql_query = result.get('sql_query', 'Query not available')
            
            # Prepare response data
            response_data = prepare_response_data(result_data)
            csv_data = format_csv_data(result_data)
            
            # Create metadata
            metadata = create_response_metadata(
                result_data, 
                has_visualization=visualization is not None
            )
            
            # Convert visualization if present
            viz_json = None
            if visualization:
                try:
                    viz_json = convert_numpy_types(json.loads(visualization.to_json()))
                except Exception as e:
                    logger.error(f"Error converting visualization: {e}")
            
            logger.info("Query processed successfully")
            
            return QueryResponse(
                success=True,
                sql_query=sql_query,
                response=base_response,
                data=response_data,
                csv_data=csv_data,
                error="",
                visualization=viz_json,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat message endpoint with voice detection and response"""
        session_id = None
        
        # Start overall timing
        total_start_time = time.time()
        timing_stats = {}
        
        try:
            # Session validation timing
            session_start = time.time()
            session_id = validate_session_id(request.session_id)
            if not session_id:
                session_id = get_or_create_session_id(None)
                
                # Create new session in MongoDB
                if self.db_manager.is_mongodb_available():
                    self.db_manager.create_chat_session(session_id)
            
            timing_stats['session_validation'] = time.time() - session_start
            
            # Input processing timing
            input_processing_start = time.time()
            
            # Determine if this is voice input and process accordingly
            detected_language = None
            question_text = None
            original_question_text = None  # Keep original question for LLM response generation
            
            if request.input_type == "voice" and request.audio_data:
                voice_processing_start = time.time()
                logger.info("Processing voice input")
                
                # Process voice input using speech service
                if self.speech_service and self.speech_service.is_available():
                    # Decode base64 audio data to bytes
                    try:
                        import base64
                        decode_start = time.time()
                        audio_bytes = base64.b64decode(request.audio_data)
                        timing_stats['audio_decode'] = time.time() - decode_start
                        logger.info(f"Decoded audio data: {len(audio_bytes)} bytes")
                        
                        stt_start = time.time()
                        voice_result = self.speech_service.speech_to_text(audio_bytes)
                        timing_stats['speech_to_text'] = time.time() - stt_start
                        logger.info(f"Speech service response: {voice_result}")
                    except Exception as decode_error:
                        logger.error(f"Error decoding audio data: {decode_error}")
                        raise Exception(f"Failed to decode audio data: {decode_error}")
                    
                    if voice_result.get("transcript"):
                        question_text = voice_result["transcript"]  # This is the English transcript
                        detected_language = voice_result.get("language_code", "en-IN")
                        logger.info(f"Voice transcription successful: {question_text} (Language: {detected_language})")
                        
                        # For voice input: If detected language is not English, we need to:
                        # 1. Translate English transcript TO the detected language for LLM (so it responds in user's language)
                        # 2. Keep English transcript for SQL processing
                        if detected_language != "en-IN" and self.speech_service.is_translation_available():
                            logger.info(f"Translating English transcript TO {detected_language} for LLM response")
                            
                            try:
                                translation_start = time.time()
                                # Translate English transcript to user's language for LLM
                                translation_result = self.speech_service.translate_text(
                                    text=question_text,
                                    source_language="en-IN",
                                    target_language=detected_language
                                )
                                timing_stats['voice_translation'] = time.time() - translation_start
                                
                                if translation_result.get("translated_text") and not translation_result.get("error"):
                                    translated_to_user_language = translation_result["translated_text"]
                                    logger.info(f"Translation successful for LLM: '{question_text}' -> '{translated_to_user_language}'")
                                    original_question_text = translated_to_user_language  # Use translated question for LLM response
                                else:
                                    logger.error(f"Translation to user language failed: {translation_result.get('error')}, using English")
                                    original_question_text = question_text  # Fallback to English
                            except Exception as translation_error:
                                logger.error(f"Translation to user language error: {translation_error}, using English")
                                original_question_text = question_text  # Fallback to English
                        else:
                            # English input or no translation service
                            original_question_text = question_text
                        
                        # question_text stays as English transcript for SQL processing
                        
                    else:
                        error_msg = voice_result.get("error", "Unknown transcription error")
                        logger.error(f"Transcription failed: {error_msg}")
                        raise Exception(f"Failed to transcribe voice input: {error_msg}")
                else:
                    raise Exception("Speech service not available")
                
                timing_stats['voice_processing'] = time.time() - voice_processing_start
            else:
                # Text input processing
                text_processing_start = time.time()
                question_text = request.question
                original_question_text = question_text  # Keep original for LLM response
                if not question_text:
                    raise Exception("No question provided")
                
                # Detect language from text if speech service is available
                if self.speech_service and self.speech_service.is_available():
                    lang_detect_start = time.time()
                    detected_language = self.speech_service.detect_language_from_text(question_text)
                    timing_stats['language_detection'] = time.time() - lang_detect_start
                    
                    # If detected language is not English, translate to English for SQL processing only
                    if detected_language and detected_language != "en-IN" and self.speech_service.is_translation_available():
                        logger.info(f"Translating text input from {detected_language} to English for SQL processing")
                        
                        try:
                            text_translation_start = time.time()
                            translation_result = self.speech_service.translate_text(
                                text=question_text,
                                source_language=detected_language,
                                target_language="en-IN"
                            )
                            timing_stats['text_translation'] = time.time() - text_translation_start
                            
                            if translation_result.get("translated_text") and not translation_result.get("error"):
                                translated_question = translation_result["translated_text"]
                                logger.info(f"Text translation successful for SQL processing: '{question_text}' -> '{translated_question}'")
                                question_text = translated_question  # Use translated question only for SQL processing
                            else:
                                logger.error(f"Text translation failed: {translation_result.get('error')}, using original question")
                                # Keep original question_text if translation fails
                        except Exception as translation_error:
                            logger.error(f"Text translation error: {translation_error}, using original question")
                            # Keep original question_text if translation fails
                    elif detected_language and detected_language != "en-IN":
                        logger.warning("Translation service not available, using original text")
                        # Keep original question_text if translation service not available
                
                timing_stats['text_processing'] = time.time() - text_processing_start
            
            timing_stats['input_processing'] = time.time() - input_processing_start
            
            logger.info(f"Processing chat query: {question_text}")
            
            # Ensure we have an original question (fallback to processed question if no translation occurred)
            if not original_question_text:
                original_question_text = question_text
            
            # Context processing timing
            context_start = time.time()
            # Get enhanced context from previous conversation
            enhanced_context = get_enhanced_chat_context(session_id, self.db_manager)
            
            # Create context-aware question if we have previous context
            contextual_question = create_context_aware_question(question_text, enhanced_context)
            timing_stats['context_processing'] = time.time() - context_start
            
            # Query processing timing
            query_processing_start = time.time()
            # Process the query using the query processor
            # Pass both translated question (for SQL) and original question (for LLM response)
            result = self.query_processor.process_user_query(
                contextual_question,  # Translated/processed question for SQL generation
                include_visualization=False,  # No visualization in chat
                original_question=original_question_text  # Original question for LLM response
            )
            timing_stats['query_processing'] = time.time() - query_processing_start
            
            if not result.get('success', False):
                raise Exception(result.get('error', 'Unknown error occurred'))
            
            result_data = result.get('data', pd.DataFrame())
            base_response = result.get('response', '')
            sql_query = result.get('sql_query', 'Query not available')
            
            # Enhanced explanation timing
            explanation_start = time.time()
            # Generate enhanced explanation using original question so it's in the right language
            explanation = await generate_enhanced_contextual_explanation(
                original_question_text, sql_query, result_data, enhanced_context, base_response, self.llm
            )
            timing_stats['explanation_generation'] = time.time() - explanation_start
            
            # Response preparation timing
            response_prep_start = time.time()
            # Prepare response data
            response_data = prepare_response_data(result_data)
            csv_data = format_csv_data(result_data)
            timing_stats['response_preparation'] = time.time() - response_prep_start
            
            # Handle audio output (no response translation needed since LLM responds in original language)
            translated_response = base_response  # LLM already responded in the original language
            audio_response = None
            
            # Generate audio response only for voice input
            if request.input_type == "voice" and detected_language and self.speech_service and self.speech_service.is_available():
                # Check if Azure TTS is available before attempting audio generation
                if self.speech_service.is_azure_tts_available():
                    audio_generation_start = time.time()
                    # Generate audio response in the detected language using the translated text
                    cleaned_translated_text = clean_md(translated_response)
                    audio_result = self.speech_service.text_to_speech(
                        cleaned_translated_text, 
                        target_language=detected_language
                    )
                    timing_stats['audio_generation'] = time.time() - audio_generation_start
                    
                    if audio_result.get("audio_data"):
                        # Convert audio bytes to base64 string
                        import base64
                        audio_response = base64.b64encode(audio_result["audio_data"]).decode('utf-8')
                        logger.info("Audio response generated successfully")
                    else:
                        logger.error(f"Audio generation failed: {audio_result.get('error', 'Unknown error')}")
                else:
                    logger.warning("Azure TTS not available - AZURE_API_KEY not configured. Skipping audio generation.")
            
            # Session management timing
            session_mgmt_start = time.time()
            # Add messages to session
            add_enhanced_message_to_session(
                session_id, question_text, base_response, 
                sql_query, result_data, explanation, self.db_manager
            )
            
            # Get chat history for response
            chat_history = get_chat_history_for_response(session_id, self.db_manager)
            timing_stats['session_management'] = time.time() - session_mgmt_start
            
            # Final response assembly timing
            response_assembly_start = time.time()
            # Create metadata
            metadata = create_response_metadata(
                result_data, 
                session_id=session_id,
                has_visualization=False
            )
            
            # Calculate total processing time
            timing_stats['total_processing'] = time.time() - total_start_time
            
            # Log performance metrics
            logger.info("Chat query processed successfully")
            logger.info(f"🕐 Performance Timing Stats: {timing_stats}")
            
            # Add timing stats to metadata for client visibility
            metadata['timing_stats'] = timing_stats
            timing_stats['response_assembly'] = time.time() - response_assembly_start
            
            return ChatResponse(
                success=True,
                session_id=session_id,
                sql_query=sql_query,
                response=base_response,
                translated_response=translated_response,
                explanation=explanation,
                data=response_data,
                csv_data=csv_data,
                error="",
                visualization=None,  # No visualization in chat
                audio_response=audio_response,
                detected_language=detected_language,
                input_type=request.input_type,
                metadata=metadata,
                chat_history=chat_history
            )
            
        except Exception as e:
            # Calculate timing even for errors
            error_total_time = time.time() - total_start_time
            timing_stats['total_processing'] = error_total_time
            
            logger.error(f"Error processing chat: {e}")
            logger.error(f"🕐 Error Processing Time: {error_total_time:.3f}s | Partial Timing: {timing_stats}")
            
            return ChatResponse(
                success=False,
                session_id=session_id or "",
                sql_query="",
                response="",
                translated_response=None,
                explanation="",
                data=[],
                csv_data="",
                error=str(e),
                visualization=None,
                audio_response=None,
                detected_language=None,
                input_type=getattr(request, 'input_type', 'text'),
                metadata={"has_error": True, "timing_stats": timing_stats},
                chat_history=[]
            )
    
    async def create_new_chat_session(self) -> SessionResponse:
        """Create new chat session endpoint"""
        try:
            # Generate new session ID
            session_id = get_or_create_session_id(None)
            
            # Create session in MongoDB if available
            if self.db_manager and self.db_manager.is_mongodb_available():
                self.db_manager.create_chat_session(session_id)
                message = "New chat session created successfully"
            else:
                message = "New session created (MongoDB not available - session will be temporary)"
            
            return SessionResponse(
                success=True,
                session_id=session_id,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_chat_sessions(self, limit: int = Query(default=50, ge=1, le=100)) -> SessionsListResponse:
        """Get all chat sessions endpoint"""
        try:
            if not self.db_manager or not self.db_manager.is_mongodb_available():
                return SessionsListResponse(
                    success=False,
                    sessions=[],
                    total_count=0
                )
            
            sessions = self.db_manager.get_all_chat_sessions(limit=limit)
            
            return SessionsListResponse(
                success=True,
                sessions=sessions,
                total_count=len(sessions)
            )
            
        except Exception as e:
            logger.error(f"Error getting chat sessions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_chat_history(self, session_id: str, limit: int = Query(default=50, ge=1, le=100)) -> ChatHistoryResponse:
        """Get chat history for a session endpoint"""
        try:
            if not self.db_manager or not self.db_manager.is_mongodb_available():
                return ChatHistoryResponse(
                    success=False,
                    session_id=session_id,
                    messages=[],
                    total_messages=0
                )
            
            messages = self.db_manager.get_chat_history(session_id, limit=limit)
            
            return ChatHistoryResponse(
                success=True,
                session_id=session_id,
                messages=messages,
                total_messages=len(messages)
            )
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def execute_sql_directly(self, sql_query: str = Query(..., description="SQL query to execute")) -> QueryResponse:
        """Direct SQL execution endpoint"""
        try:
            logger.info(f"Executing direct SQL: {sql_query}")
            
            # Execute the query
            result_data = self.db_manager.execute_query(sql_query)
            
            # Prepare response data
            response_data = prepare_response_data(result_data)
            csv_data = format_csv_data(result_data)
            
            # Create metadata
            metadata = create_response_metadata(result_data)
            
            logger.info("Direct SQL executed successfully")
            
            return QueryResponse(
                success=True,
                sql_query=sql_query,
                response=f"Query executed successfully. Returned {len(result_data)} rows.",
                data=response_data,
                csv_data=csv_data,
                error="",
                visualization=None,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def decide(self, csv_data: CSVData):
        result = decide_graph_from_string(
            csv_content=csv_data.csv_content,
            user_query=csv_data.user_query or "",
            response_text=csv_data.response_text or ""
        )
        return result