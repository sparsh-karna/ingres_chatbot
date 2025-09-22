"""
INGRES AI ChatBot - FastAPI Backend
RESTful API for querying groundwater resource data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import os
import uuid
from collections import defaultdict, deque
from sqlalchemy import text
from database_manager import DatabaseManager
from query_processor import QueryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def get_or_create_session_id(provided_session_id: Optional[str]) -> str:
    """Get existing session ID or create a new one"""
    if provided_session_id and provided_session_id in chat_sessions:
        return provided_session_id
    return str(uuid.uuid4())

def add_message_to_session(session_id: str, role: str, content: str):
    """Add a message to the chat session history"""
    message = ChatMessage(
        role=role,
        content=content,
        timestamp=datetime.now()
    )
    chat_sessions[session_id].append(message)

def get_chat_context(session_id: str) -> str:
    """Get formatted chat context for LLM"""
    if session_id not in chat_sessions or not chat_sessions[session_id]:
        return ""
    
    context = "Previous conversation context:\n"
    for message in chat_sessions[session_id]:
        context += f"{message.role.title()}: {message.content}\n"
    context += "\n"
    return context

def get_structured_chat_context(session_id: str) -> Dict:
    """Get structured chat context instead of plain text"""
    if session_id not in chat_sessions or not chat_sessions[session_id]:
        return {}
    
    context = {
        "previous_questions": [],
        "previous_responses": [],
        "conversation_flow": []
    }
    
    for message in chat_sessions[session_id]:
        if message.role == "user":
            context["previous_questions"].append(message.content)
        elif message.role == "assistant":
            context["previous_responses"].append(message.content)
        
        context["conversation_flow"].append({
            "role": message.role,
            "content": message.content[:200] + "..." if len(message.content) > 200 else message.content,
            "timestamp": message.timestamp
        })
    
    return context

def create_context_aware_question(user_question: str, enhanced_context: Dict) -> str:
    """Create a context-aware question using previous query results"""
    
    if not enhanced_context.get('previous_queries'):
        return user_question
    
    context_prompt = "CONTEXT FROM PREVIOUS CONVERSATION:\n\n"
    
    # Add information about previous queries and their results
    for i, query_info in enumerate(enhanced_context['previous_queries'][-2:], 1):  # Last 2 queries
        context_prompt += f"Previous Query {i}:\n"
        context_prompt += f"SQL: {query_info['sql_query']}\n"
        
        if query_info.get('data_summary'):
            context_prompt += f"Results: {query_info['data_summary']['rows_count']} rows\n"
            context_prompt += f"Columns: {', '.join(query_info['data_summary']['columns'])}\n"
            
            # Add sample data for context
            if query_info.get('sample_data'):
                context_prompt += "Sample results:\n"
                for row in query_info['sample_data']:
                    context_prompt += f"  {row}\n"
            
            context_prompt += "\n"
    
    # Add mentioned entities (states, districts) for reference
    if enhanced_context.get('mentioned_entities'):
        context_prompt += f"Previously mentioned: {', '.join(enhanced_context['mentioned_entities'][:10])}\n\n"
    
    # Enhanced question with proper context
    enhanced_question = f"""{context_prompt}CURRENT USER QUESTION: {user_question}

IMPORTANT INSTRUCTIONS:
1. Use the context above to understand what "these states", "those districts", "the data", etc. refer to
2. If the user is asking about previously mentioned entities, use their actual names from the context
3. Build upon previous query results when relevant
4. If referring to previous results, use specific state/district names rather than placeholders
"""
    
    return enhanced_question

async def generate_enhanced_contextual_explanation(
    question: str, 
    sql_query: str, 
    data: pd.DataFrame,
    enhanced_context: Dict,
    base_response: str
) -> str:
    """Generate explanation that has access to previous query results and context"""
    
    if not query_processor or not query_processor.llm:
        return base_response
    
    try:
        # Build context with previous data access
        context_info = ""
        if enhanced_context.get('previous_queries'):
            context_info += "PREVIOUS CONVERSATION DATA:\n"
            for i, query_info in enumerate(enhanced_context['previous_queries']):
                context_info += f"Query {i+1}: {query_info.get('data_summary', {}).get('rows_count', 0)} rows\n"
                if query_info.get('sample_data'):
                    context_info += f"Sample data: {query_info['sample_data'][0] if query_info['sample_data'] else 'None'}\n"
            context_info += "\n"
        
        # Current data summary
        data_summary = ""
        if not data.empty:
            data_summary = f"""
CURRENT QUERY RESULTS:
- {len(data)} rows returned
- Columns: {', '.join(data.columns)}
- Sample data: {data.head(2).to_dict(orient='records') if len(data) > 0 else 'No data'}
"""
        else:
            data_summary = "CURRENT QUERY RESULTS: No data returned\n"
        
        explanation_prompt = f"""
You are a helpful groundwater data analyst with access to conversation history and previous query results.

{context_info}
CURRENT QUESTION: {question}
CURRENT SQL: {sql_query}
{data_summary}
BASE RESPONSE: {base_response}

IMPORTANT CONTEXT RULES:
1. You have access to previous query results shown above
2. When user asks "these states" or similar references, you know which states from previous queries
3. If no current data is returned but user is asking about previous results, reference the previous data
4. Be specific about what you found vs. what you didn't find
5. Don't hallucinate - only reference data that actually exists in the context above

Provide a helpful, accurate response that:
1. Addresses the user's current question directly
2. Uses information from previous queries when relevant
3. Explains why no data might be found (if applicable)
4. Suggests next steps or alternative queries
5. Is conversational and helpful

Response:
"""
        
        response = query_processor.llm.invoke(explanation_prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating enhanced contextual explanation: {e}")
        return base_response

async def process_query_with_context(processor: QueryProcessor, question: str, context: Dict) -> Dict:
    """Process query with structured context awareness"""
    
    # Create context-aware prompt for better SQL generation
    context_prompt = ""
    if context.get("previous_questions"):
        context_prompt += "Previous questions in this conversation:\n"
        for i, prev_q in enumerate(context["previous_questions"][-2:], 1):  # Last 2 questions
            context_prompt += f"{i}. {prev_q}\n"
        context_prompt += "\n"
    
    if context.get("previous_responses"):
        context_prompt += "Key information from previous responses:\n"
        # Extract key information from previous responses
        for prev_resp in context["previous_responses"][-1:]:  # Last response only
            # Simple heuristic to extract key info
            if "states" in prev_resp.lower() or "districts" in prev_resp.lower():
                context_prompt += f"- {prev_resp[:100]}...\n"
        context_prompt += "\n"
    
    # Enhanced question with better context integration
    if context_prompt:
        enhanced_question = f"""Given the conversation context:
{context_prompt}
Current question: {question}

Please consider the context when generating the SQL query, but focus primarily on the current question."""
    else:
        enhanced_question = question
    
    # Process with context-aware question
    return processor.process_user_query(enhanced_question, include_visualization=False)

async def generate_contextual_explanation(
    question: str, 
    sql_query: str, 
    data: pd.DataFrame,
    context: Dict,
    base_response: str
) -> str:
    """Generate explanation that considers conversation context"""
    
    if not query_processor or not query_processor.llm:
        return base_response
    
    try:
        # Build context-aware prompt
        context_info = ""
        if context and isinstance(context, dict):
            if context.get("previous_questions"):
                context_info += f"Previous questions: {'; '.join(context['previous_questions'][-2:])}\n"
        
        data_summary = ""
        if not data.empty:
            data_summary = f"""
Query Results:
- {len(data)} rows returned
- Key columns: {', '.join(data.columns[:5])}
- Sample: {data.head(2).to_string(index=False, max_cols=3)}
"""
        
        explanation_prompt = f"""
You are a helpful assistant explaining groundwater data query results in a conversational context.

{context_info}
Current Question: {question}
Base Response: {base_response}
{data_summary}

Provide a clear, conversational explanation that:
1. Directly answers the user's question
2. Considers the conversation context (if any)
3. Highlights key insights from the data
4. Avoids repeating information already provided in base response
5. Uses a friendly, helpful tone

Keep the response concise and focused on what the user specifically asked about.
"""
        
        response = query_processor.llm.invoke(explanation_prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating contextual explanation: {e}")
        return base_response
    
def add_enhanced_message_to_session(
    session_id: str, 
    role: str, 
    content: str, 
    sql_query: Optional[str] = None,
    data: Optional[pd.DataFrame] = None
):
    """Add an enhanced message with query context to the chat session"""
    
    # Prepare data summary and limited raw data
    data_summary = None
    raw_data = None
    
    if data is not None and not data.empty:
        data_summary = {
            "rows_count": len(data),
            "columns": list(data.columns),
            "sample_values": {}
        }
        
        # Store sample values for key columns
        for col in data.columns[:5]:  # First 5 columns only
            if data[col].dtype in ['object', 'string']:
                unique_vals = data[col].dropna().unique()[:3]  # First 3 unique values
                data_summary["sample_values"][col] = list(unique_vals)
            elif data[col].dtype in ['int64', 'float64']:
                data_summary["sample_values"][col] = {
                    "min": float(data[col].min()) if not data[col].isna().all() else None,
                    "max": float(data[col].max()) if not data[col].isna().all() else None,
                    "avg": float(data[col].mean()) if not data[col].isna().all() else None
                }
        
        # Store limited raw data (first 3 rows for context)
        if len(data) > 0:
            sample_data = data.head(3).to_dict(orient='records')
            # Convert numpy types for JSON serialization
            raw_data = convert_numpy_types(sample_data)
    
    message = EnhancedChatMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        sql_query=sql_query,
        data_summary=data_summary,
        raw_data=raw_data
    )
    
    chat_sessions[session_id].append(message)

def get_enhanced_chat_context(session_id: str) -> Dict:
    """Get enhanced chat context with access to previous query results"""
    if session_id not in chat_sessions or not chat_sessions[session_id]:
        return {}
    
    context = {
        "conversation_history": [],
        "previous_queries": [],
        "available_data": {},
        "mentioned_entities": set()
    }
    
    for message in chat_sessions[session_id]:
        # Add to conversation history
        context["conversation_history"].append({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp
        })
        
        # If this is an assistant message with query results
        if message.role == "assistant" and message.sql_query:
            query_info = {
                "sql_query": message.sql_query,
                "data_summary": message.data_summary,
                "sample_data": message.raw_data
            }
            context["previous_queries"].append(query_info)
            
            # Extract mentioned entities from data
            if message.data_summary:
                for col, values in message.data_summary.get("sample_values", {}).items():
                    if col.lower() in ['state', 'district'] and isinstance(values, list):
                        context["mentioned_entities"].update(values)
                    
                # Store data structure for reference
                context["available_data"][len(context["previous_queries"])-1] = {
                    "columns": message.data_summary.get("columns", []),
                    "row_count": message.data_summary.get("rows_count", 0),
                    "sample_data": message.raw_data
                }
    
    # Convert set to list for JSON serialization
    context["mentioned_entities"] = list(context["mentioned_entities"])
    
    return context


    
def convert_dataframe_for_response(df: pd.DataFrame) -> Tuple[List[Dict], str]:
    """Convert DataFrame to proper format for JSON response"""
    # Handle data type conversion for JSON serialization
    df_converted = df.copy()
    for col in df_converted.columns:
        if df_converted[col].dtype.name.startswith('int'):
            df_converted[col] = df_converted[col].astype('Int64').astype(object).where(pd.notnull(df_converted[col]), None)
        elif df_converted[col].dtype.name.startswith('float'):
            df_converted[col] = df_converted[col].astype('float64').astype(object).where(pd.notnull(df_converted[col]), None)
        elif df_converted[col].dtype.name == 'object':
            df_converted[col] = df_converted[col].astype(str).where(pd.notnull(df_converted[col]), None)
    
    data_list = df_converted.to_dict(orient='records')
    data_list = convert_numpy_types(data_list)
    csv_data = df.to_csv(index=False)
    
    return data_list, csv_data

async def generate_explanation(question: str, sql_query: str, data: pd.DataFrame, chat_context: str = "") -> str:
    """Generate explanation using LLM"""
    if not query_processor or not query_processor.llm:
        return "Explanation generation not available."
    
    try:
        # Prepare data summary
        data_summary = ""
        if not data.empty:
            data_summary = f"""
Data Summary:
- Rows returned: {len(data)}
- Columns: {', '.join(data.columns)}
- Sample data (first 3 rows):
{data.head(3).to_string(index=False)}
"""
        
        explanation_prompt = f"""
You are an expert data analyst explaining groundwater resource data query results to users.

{chat_context}

User Question: {question}

SQL Query Used: {sql_query}

{data_summary}

Please provide a clear, conversational explanation of:
1. What the query found
2. Key insights from the data
3. What this means in practical terms for groundwater resources

Keep the explanation concise but informative, suitable for both technical and non-technical users.
"""
        
        response = query_processor.llm.invoke(explanation_prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return "Unable to generate explanation at this time."

# Initialize FastAPI app
app = FastAPI(
    title="INGRES AI ChatBot API",
    description="Intelligent Virtual Assistant for India Ground Water Resource Estimation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for database components
db_manager: Optional[DatabaseManager] = None
query_processor: Optional[QueryProcessor] = None

# Chat session storage (in production, use Redis or database)
chat_sessions = defaultdict(lambda: deque(maxlen=6))  # Store last 3 conversations per session

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about groundwater data", min_length=1)
    include_visualization: bool = Field(True, description="Whether to generate visualizations")

class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural language question about groundwater data", min_length=1)
    session_id: Optional[str] = Field(None, description="Chat session ID for context")
    include_visualization: bool = Field(True, description="Whether to generate visualizations")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")

class EnhancedChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    sql_query: Optional[str] = Field(None, description="SQL query used (for assistant messages)")
    data_summary: Optional[Dict] = Field(None, description="Summary of returned data")
    raw_data: Optional[List[Dict]] = Field(None, description="Actual query results (limited)")

class ChatResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was successful")
    session_id: str = Field(..., description="Chat session ID")
    sql_query: str = Field("", description="Generated SQL query")
    response: str = Field("", description="Natural language response")
    explanation: str = Field("", description="LLM explanation of the results")
    data: List[Dict] = Field(default_factory=list, description="Query result data")
    csv_data: str = Field("", description="Results in CSV format")
    error: str = Field("", description="Error message if any")
    visualization: Optional[Dict] = Field(None, description="Plotly visualization JSON")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Recent chat history")

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was successful")
    sql_query: str = Field("", description="Generated SQL query")
    response: str = Field("", description="Natural language response")
    data: List[Dict] = Field(default_factory=list, description="Query result data")
    error: str = Field("", description="Error message if any")
    visualization: Optional[Dict] = Field(None, description="Plotly visualization JSON")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class DatabaseInfo(BaseModel):
    table_name: str
    year: str
    column_count: int
    row_count: int

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: datetime = Field(..., description="Current server time")
    database_connected: bool = Field(..., description="Database connection status")

# Startup event to initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize database manager and query processor on startup"""
    global db_manager, query_processor
    try:
        logger.info("Initializing INGRES ChatBot components...")
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        logger.info("INGRES ChatBot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Shutdown event to cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global db_manager
    if db_manager:
        try:
            db_manager.close_connection()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and database health"""
    database_connected = False
    if db_manager:
        try:
            with db_manager.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            database_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
    
    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        timestamp=datetime.now(),
        database_connected=database_connected
    )

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return results"""
    if not query_processor:
        raise HTTPException(status_code=500, detail="Query processor not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}")
        result = query_processor.process_user_query(request.question)
        
        # Convert DataFrame to list of dictionaries with proper type conversion
        data_list = []
        if not result['data'].empty:
            # Convert numpy data types to Python native types for JSON serialization
            df_converted = result['data'].copy()
            for col in df_converted.columns:
                if df_converted[col].dtype.name.startswith('int'):
                    df_converted[col] = df_converted[col].astype('Int64').astype(object).where(pd.notnull(df_converted[col]), None)
                elif df_converted[col].dtype.name.startswith('float'):
                    df_converted[col] = df_converted[col].astype('float64').astype(object).where(pd.notnull(df_converted[col]), None)
                elif df_converted[col].dtype.name == 'object':
                    df_converted[col] = df_converted[col].astype(str).where(pd.notnull(df_converted[col]), None)
            
            data_list = df_converted.to_dict(orient='records')
            
            # Additional cleanup to ensure all values are JSON serializable
            data_list = convert_numpy_types(data_list)
        
        # Convert Plotly figure to JSON if visualization exists
        viz_json = None
        if result['visualization'] and request.include_visualization:
            try:
                viz_json = result['visualization'].to_dict()
                # Clean up numpy arrays in visualization JSON
                viz_json = convert_numpy_types(viz_json)
            except Exception as e:
                logger.warning(f"Failed to serialize visualization: {e}")
        
        # Prepare metadata
        metadata = {
            "rows_returned": len(data_list),
            "columns": list(result['data'].columns) if not result['data'].empty else [],
            "execution_time": datetime.now().isoformat(),
            "has_visualization": viz_json is not None
        }
        
        response_obj = QueryResponse(
            success=result['success'],
            sql_query=result['sql_query'],
            response=result['response'],
            data=data_list,
            error=result['error'],
            visualization=viz_json,
            metadata=metadata
        )
        
        return response_obj
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Chat endpoint with session management
@app.post("/chat", response_model=ChatResponse)
async def chat_with_enhanced_context(request: ChatRequest):
    """Enhanced chat endpoint with better context and data storage"""
    if not query_processor:
        raise HTTPException(status_code=500, detail="Query processor not initialized")
    
    try:
        # Get or create session ID
        session_id = get_or_create_session_id(request.session_id)
        
        # Add user message to session (no data for user messages)
        add_enhanced_message_to_session(session_id, "user", request.question)
        
        # Get enhanced context with previous query results
        enhanced_context = get_enhanced_chat_context(session_id)
        
        logger.info(f"Processing chat query for session {session_id}: {request.question}")
        logger.info(f"Enhanced context available: {bool(enhanced_context.get('previous_queries'))}")
        
        # Create context-aware enhanced question
        context_enhanced_question = create_context_aware_question(
            request.question, 
            enhanced_context
        )
        
        # Process the enhanced query
        result = query_processor.process_user_query(
            context_enhanced_question, 
            include_visualization=False
        )
        
        # Convert DataFrame to proper format
        data_list = []
        csv_data = ""
        
        if not result['data'].empty:
            data_list, csv_data = convert_dataframe_for_response(result['data'])
        
        # Generate contextual explanation with access to previous data
        explanation = await generate_enhanced_contextual_explanation(
            request.question,
            result['sql_query'],
            result['data'],
            enhanced_context,
            result['response']
        )
        
        # Use explanation as the main response if it's better
        final_response = explanation if explanation and len(explanation) > len(result['response']) else result['response']
        
        # Add assistant response to session WITH the query data
        add_enhanced_message_to_session(
            session_id, 
            "assistant", 
            final_response,
            sql_query=result['sql_query'],
            data=result['data']
        )
        
        # Get updated chat history (convert to original format for response)
        chat_history = []
        for msg in chat_sessions[session_id]:
            chat_history.append(ChatMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            ))
        
        # Enhanced metadata
        metadata = {
            "rows_returned": len(data_list),
            "columns": list(result['data'].columns) if not result['data'].empty else [],
            "execution_time": datetime.now().isoformat(),
            "has_visualization": False,
            "session_id": session_id,
            "context_used": bool(enhanced_context.get('previous_queries')),
            "context_queries_count": len(enhanced_context.get('previous_queries', [])),
            "mentioned_entities_count": len(enhanced_context.get('mentioned_entities', []))
        }
        
        return ChatResponse(
            success=result['success'],
            session_id=session_id,
            sql_query=result['sql_query'],
            response=result['response'],
            explanation=explanation,
            data=data_list,
            csv_data=csv_data,
            error=result['error'],
            visualization=None,
            metadata=metadata,
            chat_history=chat_history
        )
        
    except Exception as e:
        logger.error(f"Error processing enhanced chat query: {e}")
        if 'session_id' in locals():
            add_enhanced_message_to_session(session_id, "assistant", f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")



# Database info endpoint
@app.get("/database/info", response_model=List[DatabaseInfo])
async def get_database_info():
    """Get information about available database tables"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        with db_manager.engine.connect() as conn:
            # Get table information
            result = conn.execute(text("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' 
                AND table_name LIKE 'groundwater_data_%'
                ORDER BY table_name
            """))
            
            tables_info = []
            for row in result:
                table_name = row[0]
                year = table_name.replace('groundwater_data_', '').replace('_', '-')
                column_count = row[1]
                
                # Get row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.fetchone()[0]
                
                tables_info.append(DatabaseInfo(
                    table_name=table_name,
                    year=year,
                    column_count=column_count,
                    row_count=row_count
                ))
            
            return tables_info
            
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving database info: {str(e)}")

# Schema endpoint
@app.get("/database/schema/{table_name}")
async def get_table_schema(table_name: str):
    """Get schema information for a specific table"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        schema = db_manager.get_table_schema(table_name)
        return {"table_name": table_name, "schema": schema}
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving schema: {str(e)}")

# Sample questions endpoint
@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions users can ask"""
    sample_questions = [
        "What are the top 10 districts with highest groundwater recharge in 2024-2025?",
        "Show me groundwater extraction data for Maharashtra",
        "Which states have critical groundwater extraction levels?",
        "Compare rainfall patterns between 2020 and 2024",
        "What is the average groundwater recharge in Rajasthan?",
        "Show me districts with over-exploited groundwater resources",
        "Plot groundwater recharge trends for Maharashtra", 
        "Show recharge hierarchy by state and district",
        "Visualize rainfall vs. recharge in Karnataka",
        "Which is better - rice or wheat for growing in Chennai?",
        "Top 5 states with safest groundwater levels",
        "Districts in Tamil Nadu with highest rainfall in 2023-2024"
    ]
    
    return {
        "sample_questions": sample_questions,
        "tips": [
            "Be specific about years (e.g., '2024-2025')",
            "Mention specific states or districts",
            "Ask about specific metrics like rainfall, recharge, extraction",
            "Use comparative language for trends",
            "Include 'plot', 'chart', or 'visualize' for visualizations"
        ]
    }

# Download data endpoint
@app.post("/download")
async def download_query_data(request: QueryRequest):
    """Execute query and return downloadable CSV"""
    if not query_processor:
        raise HTTPException(status_code=500, detail="Query processor not initialized")
    
    try:
        result = query_processor.process_user_query(request.question)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        if result['data'].empty:
            raise HTTPException(status_code=404, detail="No data found for the query")
        
        # Create temporary CSV file
        filename = f"ingres_query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = f"/tmp/{filename}"
        
        result['data'].to_csv(filepath, index=False)
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating download: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

# Direct SQL execution endpoint (with safety checks)
@app.post("/execute-sql")
async def execute_sql(sql_query: str = Query(..., description="SQL query to execute")):
    """Execute SQL query directly (with safety validation)"""
    if not query_processor:
        raise HTTPException(status_code=500, detail="Query processor not initialized")
    
    try:
        # Use the safe execution method
        success, data, message = query_processor.execute_query_safely(sql_query)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Convert DataFrame with proper type handling
        data_list = []
        if not data.empty:
            # Convert numpy data types to Python native types
            df_converted = data.copy()
            for col in df_converted.columns:
                if df_converted[col].dtype.name.startswith('int'):
                    df_converted[col] = df_converted[col].astype('Int64').astype(object).where(pd.notnull(df_converted[col]), None)
                elif df_converted[col].dtype.name.startswith('float'):
                    df_converted[col] = df_converted[col].astype('float64').astype(object).where(pd.notnull(df_converted[col]), None)
                elif df_converted[col].dtype.name == 'object':
                    df_converted[col] = df_converted[col].astype(str).where(pd.notnull(df_converted[col]), None)
            
            data_list = df_converted.to_dict(orient='records')
            
            # Additional cleanup for JSON serialization
            data_list = convert_numpy_types(data_list)
        
        return {
            "success": True,
            "data": data_list,
            "rows_returned": len(data_list),
            "columns": list(data.columns) if not data.empty else [],
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {str(e)}")

# Chat session management endpoints
@app.post("/chat/new-session")
async def create_new_chat_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = deque(maxlen=3)
    return {
        "success": True,
        "session_id": session_id,
        "message": "New chat session created"
    }

@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session history"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = list(chat_sessions[session_id])
    return {
        "success": True,
        "session_id": session_id,
        "chat_history": history,
        "message_count": len(history)
    }

@app.delete("/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear chat session history"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_sessions[session_id].clear()
    return {
        "success": True,
        "session_id": session_id,
        "message": "Chat session cleared"
    }

@app.get("/chat/sessions")
async def list_active_sessions():
    """List all active chat sessions"""
    active_sessions = []
    for session_id, messages in chat_sessions.items():
        if messages:  # Only include sessions with messages
            last_message = messages[-1]
            active_sessions.append({
                "session_id": session_id,
                "message_count": len(messages),
                "last_activity": last_message.timestamp,
                "last_message_preview": last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
            })
    
    return {
        "success": True,
        "active_sessions": active_sessions,
        "total_sessions": len(active_sessions)
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "INGRES AI ChatBot API",
        "description": "Intelligent Virtual Assistant for India Ground Water Resource Estimation System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /query": "Process natural language queries",
            "POST /chat": "Chat with context and session management",
            "POST /chat/new-session": "Create new chat session",
            "GET /chat/session/{session_id}": "Get chat session history",
            "DELETE /chat/session/{session_id}": "Clear chat session",
            "GET /chat/sessions": "List active chat sessions",
            "GET /database/info": "Get database table information",
            "GET /database/schema/{table_name}": "Get table schema",
            "GET /sample-questions": "Get sample questions and tips",
            "POST /download": "Download query results as CSV",
            "POST /execute-sql": "Execute SQL directly"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )