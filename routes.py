"""
Route Handlers for INGRES ChatBot FastAPI Backend
"""

import json
import logging
import re
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
        try:
            # Validate and get session ID
            session_id = validate_session_id(request.session_id)
            if not session_id:
                session_id = get_or_create_session_id(None)
                
                # Create new session in MongoDB
                if self.db_manager.is_mongodb_available():
                    self.db_manager.create_chat_session(session_id)
            
            # Determine if this is voice input and process accordingly
            detected_language = None
            question_text = None
            
            if request.input_type == "voice" and request.audio_data:
                logger.info("Processing voice input")
                
                # Process voice input using speech service
                if self.speech_service and self.speech_service.is_available():
                    # Decode base64 audio data to bytes
                    try:
                        import base64
                        audio_bytes = base64.b64decode(request.audio_data)
                        logger.info(f"Decoded audio data: {len(audio_bytes)} bytes")
                        
                        voice_result = self.speech_service.speech_to_text(audio_bytes)
                        logger.info(f"Speech service response: {voice_result}")
                    except Exception as decode_error:
                        logger.error(f"Error decoding audio data: {decode_error}")
                        raise Exception(f"Failed to decode audio data: {decode_error}")
                    
                    if voice_result.get("transcript"):
                        question_text = voice_result["transcript"]
                        detected_language = voice_result.get("language_code", "en-IN")
                        logger.info(f"Voice transcription successful: {question_text} (Language: {detected_language})")
                        
                        # If detected language is not English, translate to English using LLM
                        if detected_language != "en-IN":
                            logger.info(f"Translating question from {detected_language} to English using LLM")
                            
                            # Create language mapping for better LLM understanding
                            language_map = {
                                "hi-IN": "Hindi", "gu-IN": "Gujarati", "mr-IN": "Marathi", 
                                "bn-IN": "Bengali", "ta-IN": "Tamil", "te-IN": "Telugu",
                                "kn-IN": "Kannada", "ml-IN": "Malayalam", "pa-IN": "Punjabi",
                                "or-IN": "Odia", "as-IN": "Assamese"
                            }
                            source_language_name = language_map.get(detected_language, detected_language)
                            
                            # Use LLM to translate the question to English
                            translation_prompt = f"""Please translate the following text to {source_language_name} . 
                            Only return the translation, no additional text or explanations.

{source_language_name} text to translate: {question_text}"""

                            try:
                                llm_translation = await self.llm.ainvoke(translation_prompt)
                                if hasattr(llm_translation, 'content'):
                                    translated_question = llm_translation.content.strip()
                                else:
                                    translated_question = str(llm_translation).strip()
                                
                                logger.info(f"LLM translation successful: '{question_text}' -> '{translated_question}'")
                                question_text = translated_question  # Use translated question for processing
                            except Exception as llm_error:
                                logger.error(f"LLM translation failed: {llm_error}, using original question")
                                # Keep original question_text if translation fails
                        
                    else:
                        error_msg = voice_result.get("error", "Unknown transcription error")
                        logger.error(f"Transcription failed: {error_msg}")
                        raise Exception(f"Failed to transcribe voice input: {error_msg}")
                else:
                    raise Exception("Speech service not available")
            else:
                # Text input
                question_text = request.question
                if not question_text:
                    raise Exception("No question provided")
                
                # Detect language from text if speech service is available
                if self.speech_service and self.speech_service.is_available():
                    detected_language = self.speech_service.detect_language_from_text(question_text)
                
            logger.info(f"Processing chat query: {question_text}")
            
            # Get enhanced context from previous conversation
            enhanced_context = get_enhanced_chat_context(session_id, self.db_manager)
            
            # Create context-aware question if we have previous context
            contextual_question = create_context_aware_question(question_text, enhanced_context)
            
            # Process the query using the query processor
            result = self.query_processor.process_user_query(
                contextual_question, 
                include_visualization=False  # No visualization in chat
            )
            
            if not result.get('success', False):
                raise Exception(result.get('error', 'Unknown error occurred'))
            
            result_data = result.get('data', pd.DataFrame())
            base_response = result.get('response', '')
            sql_query = result.get('sql_query', 'Query not available')
            
            # Generate enhanced explanation
            explanation = await generate_enhanced_contextual_explanation(
                question_text, sql_query, result_data, enhanced_context, base_response, self.llm
            )
            
            # Prepare response data
            response_data = prepare_response_data(result_data)
            csv_data = format_csv_data(result_data)
            
            # Handle voice output if input was voice
            translated_response = base_response
            audio_response = None
            
            if request.input_type == "voice" and detected_language and self.speech_service and self.speech_service.is_available():
                # If detected language is not English, use LLM to translate the response back
                if detected_language != "en-IN":
                    logger.info(f"Using LLM to translate response back to detected language: {detected_language}")
                    
                    # Create language mapping for better LLM understanding
                    language_map = {
                        "hi-IN": "Hindi", "gu-IN": "Gujarati", "mr-IN": "Marathi", 
                        "bn-IN": "Bengali", "ta-IN": "Tamil", "te-IN": "Telugu",
                        "kn-IN": "Kannada", "ml-IN": "Malayalam", "pa-IN": "Punjabi",
                        "or-IN": "Odia", "as-IN": "Assamese"
                    }
                    target_language_name = language_map.get(detected_language, detected_language)
                    
                    # Use LLM to translate the response back to user's language
                    translation_prompt = f"""Please translate the following English text to {target_language_name}. 
                    Only return the translation, no additional text or explanations.

English text to translate: {base_response}"""

                    try:
                        llm_translation = await self.llm.ainvoke(translation_prompt)
                        if hasattr(llm_translation, 'content'):
                            translated_response = llm_translation.content.strip()
                        else:
                            translated_response = str(llm_translation).strip()
                        logger.info(f"LLM response translation successful to {target_language_name}")
                    except Exception as llm_error:
                        logger.error(f"LLM response translation failed: {llm_error}, using original response")
                        translated_response = base_response
                else:
                    translated_response = base_response
                
                # Generate audio response in the detected language using the translated text
                cleaned_translated_text = clean_md(translated_response)
                audio_result = self.speech_service.text_to_speech(
                    cleaned_translated_text, 
                    target_language=detected_language
                )
                
                if audio_result.get("audio_data"):
                    # Convert audio bytes to base64 string
                    import base64
                    audio_response = base64.b64encode(audio_result["audio_data"]).decode('utf-8')
                    logger.info("Audio response generated successfully")
                else:
                    logger.error(f"Audio generation failed: {audio_result.get('error', 'Unknown error')}")
            
            # Add messages to session
            add_enhanced_message_to_session(
                session_id, question_text, base_response, 
                sql_query, result_data, explanation, self.db_manager
            )
            
            # Get chat history for response
            chat_history = get_chat_history_for_response(session_id, self.db_manager)
            
            # Create metadata
            metadata = create_response_metadata(
                result_data, 
                session_id=session_id,
                has_visualization=False
            )
            
            logger.info("Chat query processed successfully")
            
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
            logger.error(f"Error processing chat: {e}")
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
                metadata={"has_error": True},
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
        srno, data = decide_graph_from_string(csv_data.csv_content)
        return {"srno": srno, "jsonData": data}

    async def forecast(self, csv_data: CSVData, all_col: Boolean, yrs: Number):
        df = pd.read_csv(StringIO(csv_data.csv_content))
        forecast_csv = forecast_data(df, all_col, yrs)
        #forecast_json = pd.read_csv(StringIO(forecast_csv)).to_dict(orient="records")

        return {"csvData": forecast_csv}