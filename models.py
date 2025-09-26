"""
Pydantic Models for INGRES ChatBot FastAPI Backend
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for SQL queries"""
    question: str = Field(..., description="Natural language question")
    include_visualization: bool = Field(default=True, description="Whether to generate visualization")


class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    question: Optional[str] = Field(default=None, description="Natural language question (for text input)")
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio data (for voice input)")
    input_type: str = Field(default="text", description="Input type: 'text' or 'voice'")
    session_id: Optional[str] = Field(default=None, description="Chat session ID")
    include_visualization: bool = Field(default=False, description="Whether to generate visualization")


class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    metadata: Optional[Dict] = Field(default=None, description="Additional message metadata")


class ChatSession(BaseModel):
    """Model for chat sessions"""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Number of messages in session")
    metadata: Optional[Dict] = Field(default=None, description="Session metadata")
    is_active: bool = Field(default=True, description="Whether session is active")


class QueryResponse(BaseModel):
    """Response model for query results"""
    success: bool = Field(..., description="Whether the query was successful")
    sql_query: str = Field(..., description="Generated SQL query")
    response: str = Field(..., description="Natural language response")
    data: List[Dict[str, Any]] = Field(..., description="Query results data")
    csv_data: str = Field(..., description="Results in CSV format")
    error: str = Field(default="", description="Error message if any")
    visualization: Optional[Dict] = Field(default=None, description="Plotly visualization JSON")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Chat session ID")
    sql_query: str = Field(..., description="Generated SQL query")
    response: str = Field(..., description="Natural language response")
    translated_response: Optional[str] = Field(default=None, description="Response in user's language (for voice input)")
    explanation: str = Field(..., description="LLM explanation of the results")
    data: List[Dict[str, Any]] = Field(..., description="Query results data")
    csv_data: str = Field(..., description="Results in CSV format")
    error: str = Field(default="", description="Error message if any")
    visualization: Optional[Dict] = Field(default=None, description="Plotly visualization JSON")
    audio_response: Optional[str] = Field(default=None, description="Base64 encoded audio response (for voice input)")
    detected_language: Optional[str] = Field(default=None, description="Detected input language")
    input_type: str = Field(default="text", description="Type of input processed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Recent chat history")


class SessionResponse(BaseModel):
    """Response model for session operations"""
    success: bool = Field(..., description="Whether the operation was successful")
    session_id: str = Field(..., description="Session ID")
    message: str = Field(default="", description="Success or error message")


class SessionsListResponse(BaseModel):
    """Response model for listing sessions"""
    success: bool = Field(..., description="Whether the request was successful")
    sessions: List[ChatSession] = Field(..., description="List of chat sessions")
    total_count: int = Field(..., description="Total number of sessions")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history"""
    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    total_messages: int = Field(..., description="Total number of messages in session")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    database_status: str = Field(..., description="Database connection status")
    mongodb_status: str = Field(..., description="MongoDB connection status")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(default=None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")

class CSVData(BaseModel):
    csv_content: str
    user_query: Optional[str] = ""
    response_text: Optional[str] = ""

class CSVForecastDataInput(BaseModel):
    csv_content: str
    all_col: bool = True
    yrs: int = 5

class EDADataInput(BaseModel):
    csv_content: str
    user_query: str


class GeneralChatRequest(BaseModel):
    """Request model for general chat interactions"""
    question: str = Field(..., description="General question about the groundwater dataset")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for conversation tracking")


class GeneralChatResponse(BaseModel):
    """Response model for general chat interactions"""
    success: bool = Field(..., description="Whether the request was successful")
    response: str = Field(..., description="AI response to the question")
    session_id: Optional[str] = Field(default=None, description="Session ID if provided")
    error: str = Field(default="", description="Error message if any")
    metadata: Optional[Dict] = Field(default=None, description="Additional response metadata")