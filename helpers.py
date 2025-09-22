"""
Helper Functions for INGRES ChatBot FastAPI Backend
"""

import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
from database_manager import DatabaseManager
from query_processor import QueryProcessor
import google.generativeai as genai
import csv
import io

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


def get_or_create_session_id(provided_session_id: Optional[str]) -> str:
    """Get existing session ID or create new one"""
    if provided_session_id and provided_session_id.strip():
        return provided_session_id.strip()
    return str(uuid.uuid4())


def add_message_to_session(session_id: str, role: str, content: str, 
                          sql_query: str = None, csv_data: str = None, db_manager: DatabaseManager = None):
    """Add message to chat session"""
    if db_manager and db_manager.is_mongodb_available():
        db_manager.add_chat_message(
            session_id=session_id,
            role=role,
            content=content,
            sql_query=sql_query,
            csv_data=csv_data
        )
    else:
        logger.warning("MongoDB not available, message not stored")


def get_chat_context(session_id: str, db_manager: DatabaseManager = None) -> str:
    """Get chat context as plain text"""
    if not db_manager or not db_manager.is_mongodb_available():
        return ""
    
    messages = db_manager.get_recent_context(session_id, max_messages=6)
    if not messages:
        return ""
    
    context = "Previous conversation:\n"
    for message in messages:
        context += f"{message['role']}: {message['content'][:200]}...\n"
    context += "\n"
    return context


def get_structured_chat_context(session_id: str, db_manager: DatabaseManager = None) -> Dict:
    """Get structured chat context instead of plain text"""
    if not db_manager or not db_manager.is_mongodb_available():
        return {"previous_questions": [], "previous_responses": [], "conversation_flow": []}
    
    messages = db_manager.get_recent_context(session_id, max_messages=6)
    if not messages:
        return {"previous_questions": [], "previous_responses": [], "conversation_flow": []}
    
    context = {
        "previous_questions": [],
        "previous_responses": [],
        "conversation_flow": []
    }
    
    for message in messages:
        if message["role"] == "user":
            context["previous_questions"].append(message["content"])
        elif message["role"] == "assistant":
            context["previous_responses"].append(message["content"])
        
        context["conversation_flow"].append({
            "role": message["role"],
            "content": message["content"][:200] + "..." if len(message["content"]) > 200 else message["content"],
            "timestamp": message["timestamp"]
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
    base_response: str,
    llm: ChatOpenAI
) -> str:
    """Generate enhanced explanation using context and LLM"""
    
    try:
        # Prepare context summary
        context_summary = ""
        if enhanced_context.get('previous_queries'):
            context_summary = "Previous queries in this conversation:\n"
            for i, pq in enumerate(enhanced_context['previous_queries'][-2:], 1):
                context_summary += f"{i}. {pq.get('question', 'Previous question')}\n"
        
        # Prepare data summary
        data_summary = f"""
Data Analysis Results:
- Records found: {len(data)}
- Columns: {', '.join(data.columns)}
"""
        
        if not data.empty:
            # Add top few rows as examples
            data_summary += "\nSample data:\n"
            for _, row in data.head(3).iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notnull(val)])
                data_summary += f"- {row_str}\n"
        
        explanation_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data_summary", "context_summary", "base_response"],
            template="""
You are an expert agricultural data analyst explaining groundwater resource analysis results.

User Question: {question}

SQL Query Used: {sql_query}

{data_summary}

{context_summary}

Base Response: {base_response}

Please provide a comprehensive explanation that:
1. Explains what the data shows in simple terms
2. Highlights key findings and patterns
3. Relates the results to agricultural water management
4. Mentions any limitations or considerations
5. Connects to the conversation context if relevant
6. Uses clear, non-technical language

Keep the explanation informative but accessible to non-technical users.
"""
        )
        
        chain = explanation_prompt | llm
        response = chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "data_summary": data_summary,
            "context_summary": context_summary,
            "base_response": base_response
        })
        
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating enhanced explanation: {e}")
        return base_response  # Fallback to base response


def add_enhanced_message_to_session(session_id: str, question: str, response: str, 
                                   sql_query: str, data: pd.DataFrame, 
                                   explanation: str, db_manager: DatabaseManager = None):
    """Add enhanced messages to chat session with structured data"""
    if not db_manager or not db_manager.is_mongodb_available():
        return
    
    try:
        # Add user message
        db_manager.add_chat_message(
            session_id=session_id,
            role="user",
            content=question,
            metadata={"message_type": "question"}
        )
        
        # Add assistant response with full context
        csv_data = data.to_csv(index=False) if not data.empty else ""
        
        assistant_metadata = {
            "message_type": "response",
            "sql_query": sql_query,
            "data_summary": {
                "rows_count": len(data),
                "columns": list(data.columns),
                "has_data": not data.empty
            },
            "explanation": explanation
        }
        
        db_manager.add_chat_message(
            session_id=session_id,
            role="assistant", 
            content=response,
            metadata=assistant_metadata,
            sql_query=sql_query,
            csv_data=csv_data
        )
        
    except Exception as e:
        logger.error(f"Error adding enhanced messages to session: {e}")


def get_enhanced_chat_context(session_id: str, db_manager: DatabaseManager = None) -> Dict:
    """Get enhanced chat context with query history and entity extraction"""
    if not db_manager or not db_manager.is_mongodb_available():
        return {"previous_queries": [], "mentioned_entities": [], "conversation_summary": ""}
    
    try:
        # Get recent messages with metadata
        messages = db_manager.get_recent_context(session_id, max_messages=8)
        
        enhanced_context = {
            "previous_queries": [],
            "mentioned_entities": set(),
            "conversation_summary": ""
        }
        
        for message in messages:
            if message["role"] == "assistant" and message.get("sql_query"):
                # Extract query information
                metadata = message.get("metadata", {})
                query_info = {
                    "question": "",  # Will be filled from previous user message
                    "sql_query": message["sql_query"],
                    "data_summary": metadata.get("data_summary", {}),
                    "explanation": metadata.get("explanation", "")
                }
                
                # Add sample data if available
                if message.get("csv_data") and len(message["csv_data"]) > 0:
                    try:
                        import io
                        df_sample = pd.read_csv(io.StringIO(message["csv_data"]))
                        if not df_sample.empty:
                            query_info["sample_data"] = df_sample.head(2).to_dict('records')
                    except:
                        pass
                
                enhanced_context["previous_queries"].append(query_info)
                
                # Extract entities (simple keyword extraction)
                content = message["content"].lower()
                # Look for state/district names (you could make this more sophisticated)
                common_entities = ["maharashtra", "gujarat", "rajasthan", "punjab", "haryana", 
                                 "uttar pradesh", "madhya pradesh", "bihar", "west bengal",
                                 "tamil nadu", "karnataka", "andhra pradesh", "telangana"]
                
                for entity in common_entities:
                    if entity in content:
                        enhanced_context["mentioned_entities"].add(entity.title())
        
        # Convert set to list for JSON serialization
        enhanced_context["mentioned_entities"] = list(enhanced_context["mentioned_entities"])
        
        return enhanced_context
        
    except Exception as e:
        logger.error(f"Error getting enhanced chat context: {e}")
        return {"previous_queries": [], "mentioned_entities": [], "conversation_summary": ""}


def prepare_response_data(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare DataFrame for JSON response"""
    if data.empty:
        return []
    
    # Convert DataFrame to list of dictionaries with proper type conversion
    result = []
    for _, row in data.iterrows():
        row_dict = {}
        for col, val in row.items():
            row_dict[col] = convert_numpy_types(val)
        result.append(row_dict)
    
    return result


def create_response_metadata(data: pd.DataFrame, session_id: str = None, 
                           has_visualization: bool = False) -> Dict[str, Any]:
    """Create response metadata"""
    return {
        "rows_returned": len(data),
        "columns": list(data.columns),
        "execution_time": datetime.now().isoformat(),
        "has_visualization": has_visualization,
        "session_id": session_id,
        "context_used": session_id is not None
    }


def format_csv_data(data: pd.DataFrame) -> str:
    """Format DataFrame as CSV string"""
    if data.empty:
        return ""
    return data.to_csv(index=False)


def get_chat_history_for_response(session_id: str, db_manager: DatabaseManager = None) -> List[Dict]:
    """Get formatted chat history for API response"""
    if not db_manager or not db_manager.is_mongodb_available():
        return []
    
    try:
        messages = db_manager.get_recent_context(session_id, max_messages=6)
        formatted_messages = []
        
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            })
        
        return formatted_messages
        
    except Exception as e:
        logger.error(f"Error getting chat history for response: {e}")
        return []


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    """Validate and clean session ID"""
    if not session_id:
        return None
    
    session_id = session_id.strip()
    if not session_id or session_id.lower() == 'null':
        return None
    
    return session_id


def create_error_response(error_message: str, session_id: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": error_message,
        "session_id": session_id or "",
        "sql_query": "",
        "response": "",
        "explanation": "",
        "data": [],
        "csv_data": "",
        "visualization": None,
        "metadata": {
            "execution_time": datetime.now().isoformat(),
            "has_error": True
        },
        "chat_history": []
    }

def decide_graph_from_string(csv_content: str):
    genai.configure(api_key="AIzaSyCzf-iCpO6ZV9G48b5fLB4XyfvhL4ReP3U")
    # Read CSV string using StringIO
    f = io.StringIO(csv_content)
    reader = csv.DictReader(f)
    data = [row for row in reader]

    if not data:
        return None, []

    head_sample = data[:20]
    schema = [
        {"name": key, "sampleValues": list({row[key] for row in data if row[key]})[:5]}
        for key in data[0].keys()
    ]

    viz_options = [
        {"srno": 3, "name": "Bar 3", "description": "This chart is a powerful tool for comparing multiple data series across a set of categories..."},
        {"srno": 6, "name": "Bar 6", "description": "A bar chart. Each bar represents a category, and its length or height corresponds..."},
        {"srno": 1, "name": "Composition 1", "description": "This is a multi-line chart, which plots multiple data series on a single graph..."},
        {"srno": 7, "name": "Composition 7", "description": "A composition chart, used to display continuous data over a specified period..."},
        {"srno": 4, "name": "Donut 4", "description": "A donut chart is a variation of a pie chart with a circular hole in the center..."},
    ]

    prompt = f"""
You are an AI agent that chooses the best visualization type for a dataset.
Return ONLY the srno (number) of the best chart.

Schema:
{json.dumps(schema, indent=2)}

Sample rows:
{json.dumps(head_sample, indent=2)}

Visualization options:
{json.dumps(viz_options, indent=2)}
"""

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt).text

    try:
        srno = int(response.strip())
    except ValueError:
        srno = None

    return srno, data