"""
Route Handlers for INGRES ChatBot FastAPI Backend
"""

import json
import logging
import pandas as pd
import base64
from fastapi import HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
from io import BytesIO

from models import (
    QueryRequest, ChatRequest, QueryResponse, ChatResponse, 
    SessionResponse, SessionsListResponse, ChatHistoryResponse, 
    HealthResponse, ErrorResponse
)
from helpers import (
    convert_numpy_types, get_or_create_session_id, add_message_to_session,
    get_structured_chat_context, create_context_aware_question,
    generate_enhanced_contextual_explanation, add_enhanced_message_to_session,
    get_enhanced_chat_context, prepare_response_data, create_response_metadata,
    format_csv_data, get_chat_history_for_response, validate_session_id,
    create_error_response
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
        """Process chat message endpoint"""
        try:
            # Process input based on type (text or voice)
            if request.input_type == "voice":
                if not request.audio_data:
                    raise HTTPException(status_code=400, detail="Audio data is required for voice input")
                
                if not self.speech_service or not self.speech_service.is_available():
                    raise HTTPException(status_code=503, detail="Speech service is not available")
                
                # Decode audio data
                try:
                    audio_bytes = base64.b64decode(request.audio_data)
                    audio_io = BytesIO(audio_bytes)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")
                
                # Convert speech to text
                stt_result = self.speech_service.speech_to_text(audio_io)
                if stt_result["error"]:
                    raise HTTPException(status_code=500, detail=f"Speech to text failed: {stt_result['error']}")
                
                question_text = stt_result["transcript"]
                detected_language = stt_result["language_code"]
                
                if not question_text:
                    raise HTTPException(status_code=400, detail="Could not transcribe audio")
                
                logger.info(f"Processing voice chat query: {question_text} (language: {detected_language})")
            
            else:  # text input
                if not request.question:
                    raise HTTPException(status_code=400, detail="Question is required for text input")
                
                question_text = request.question
                detected_language = None
                
                # Detect language if speech service is available
                if self.speech_service and self.speech_service.is_available():
                    detected_language = self.speech_service.detect_language_from_text(question_text)
                
                logger.info(f"Processing text chat query: {question_text}")
            
            # Validate and get session ID
            session_id = validate_session_id(request.session_id)
            if not session_id:
                session_id = get_or_create_session_id(None)
                
                # Create new session in MongoDB
                if self.db_manager.is_mongodb_available():
                    self.db_manager.create_chat_session(session_id)
            
            # Get enhanced context from previous conversation
            enhanced_context = get_enhanced_chat_context(session_id, self.db_manager)
            
            # Create context-aware question if we have previous context
            contextual_question = create_context_aware_question(question_text, enhanced_context)
            
            # Process the query (without visualization for chat)
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
            
            # Process multilingual response
            translated_response = base_response
            audio_response_data = None
            
            if self.speech_service and self.speech_service.is_available() and detected_language:
                multilingual_output = self.speech_service.process_multilingual_chat_output(
                    response_text=base_response,
                    target_language=detected_language,
                    input_type=request.input_type,
                    translate_to_english=(detected_language != "en-IN")
                )
                
                if not multilingual_output.get("error"):
                    translated_response = multilingual_output.get("translated_response", base_response)
                    if multilingual_output.get("audio_response"):
                        # Encode audio response as base64
                        audio_response_data = base64.b64encode(multilingual_output["audio_response"]).decode('utf-8')
                else:
                    logger.warning(f"Multilingual processing failed: {multilingual_output['error']}")
            
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
                audio_response=audio_response_data,
                detected_language=detected_language,
                metadata=metadata,
                chat_history=chat_history
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            error_response = create_error_response(str(e), session_id)
            return ChatResponse(**error_response)
    
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