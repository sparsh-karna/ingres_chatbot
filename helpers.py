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
import re
import io
import torch
from dotenv import load_dotenv
import os
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from custom_models import GRUPredictor

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
7. The explaination should be in the same language as the user question.

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

def decide_graph_from_string(csv_content: str, user_query: str = "", response_text: str = ""):
    """
    Analyze CSV data and recommend appropriate chart types based on user query and response context.
    
    Args:
        csv_content: CSV data as string
        user_query: User's original question
        response_text: AI generated response text
    
    Returns:
        dict: JSON with number_of_appropriate_graphs and graph_indices
    """
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyDql8hpgkDIVCT38ry8lGXp1fdxd3FwkRs")
    
    # Read CSV string using StringIO
    f = io.StringIO(csv_content)
    reader = csv.DictReader(f)
    data = [row for row in reader]

    if not data:
        return {"number_of_appropriate_graphs": 0, "graph_indices": []}

    head_sample = data[:10]
    total_entries = len(data)
    total_columns = len(data[0].keys())
    
    schema = [
        {"name": key, "type": "numeric" if any(str(row[key]).replace('.', '').replace('-', '').isdigit() for row in data[:5] if row[key]) else "categorical", 
         "sample_values": list({row[key] for row in data if row[key]})[:3]}
        for key in data[0].keys()
    ]
    
    dataset_info = {
        "total_entries": total_entries,
        "total_columns": total_columns,
        "numeric_columns": sum(1 for col in schema if col['type'] == 'numeric'),
        "categorical_columns": sum(1 for col in schema if col['type'] == 'categorical'),
        "data_size": "small" if total_entries < 50 else "medium" if total_entries < 500 else "large"
    }

    # Complete chart options with code requirements
    chart_options = [
        {
            "index": 0,
            "name": "Bar Chart - Active",
            "description": "Single bar chart with active/highlighted bars for categorical comparisons",
            "code_structure": {
                "data_format": "[{category: 'chrome', value: 187, fill: 'var(--color-chrome)'}, ...]",
                "required_fields": ["category (string)", "value (numeric)", "fill (optional)"],
                "recharts_config": "BarChart with Bar dataKey='value', activeIndex for highlighting",
                "card_elements": "CardTitle, CardDescription, CardFooter with trends"
            },
            "data_requirements": {
                "min_columns": 2,
                "categorical_cols": 1,
                "numeric_cols": 1,
                "ideal_rows": "5-20",
                "example": "States by groundwater level, Districts by rainfall"
            }
        },
        {
            "index": 1,
            "name": "Bar Chart - Stacked",
            "description": "Stacked bar chart showing composition across categories with multiple data series",
            "code_structure": {
                "data_format": "[{category: 'January', series1: 186, series2: 80}, ...]",
                "required_fields": ["category (string)", "2+ numeric series"],
                "recharts_config": "BarChart with multiple Bar components, stackId='a', ChartLegend",
                "card_elements": "CardTitle, CardDescription, Legend, CardFooter"
            },
            "data_requirements": {
                "min_columns": 3,
                "categorical_cols": 1,
                "numeric_cols": "2+",
                "ideal_rows": "3-15",
                "example": "Years by (rainfall + extraction), States by (recharge + usage)"
            }
        },
        {
            "index": 2,
            "name": "Pie Chart - Donut",
            "description": "Donut pie chart with center text showing proportional data",
            "code_structure": {
                "data_format": "[{category: 'chrome', value: 275, fill: 'var(--color-chrome)'}, ...]",
                "required_fields": ["category (string)", "value (numeric)", "fill (required)"],
                "recharts_config": "PieChart with Pie, innerRadius=60, Label in center",
                "card_elements": "CardTitle, CardDescription, Center total, CardFooter"
            },
            "data_requirements": {
                "min_columns": 2,
                "categorical_cols": 1,
                "numeric_cols": 1,
                "ideal_rows": "3-8 (max for readability)",
                "example": "Groundwater stages distribution, State-wise water usage"
            }
        },
        {
            "index": 3,
            "name": "Line Chart - Dots",
            "description": "Line chart with colored dots for trend analysis",
            "code_structure": {
                "data_format": "[{sequence: 'chrome', value: 275, fill: 'var(--color-chrome)'}, ...]",
                "required_fields": ["sequence (string/ordinal)", "value (numeric)", "fill (per dot)"],
                "recharts_config": "LineChart with Line, custom Dot components with individual colors",
                "card_elements": "CardTitle, CardDescription, CardFooter with insights"
            },
            "data_requirements": {
                "min_columns": 2,
                "categorical_cols": "1 (ordinal/sequential)",
                "numeric_cols": 1,
                "ideal_rows": "5-30",
                "example": "Monthly trends, Year-over-year changes"
            }
        },
        {
            "index": 4,
            "name": "Line Chart - Multiple",
            "description": "Multiple line series comparison over time periods",
            "code_structure": {
                "data_format": "[{period: 'January', series1: 186, series2: 80}, ...]",
                "required_fields": ["period (string)", "2+ numeric series for lines"],
                "recharts_config": "LineChart with multiple Line components, different colors",
                "card_elements": "CardTitle, CardDescription, Multiple lines, CardFooter with comparison"
            },
            "data_requirements": {
                "min_columns": 3,
                "categorical_cols": "1 (time periods)",
                "numeric_cols": "2+",
                "ideal_rows": "6-24",
                "example": "Compare rainfall vs extraction by month, Multi-state trends"
            }
        }
    ]

    prompt = f"""
You are an expert data visualization analyst and React developer. Analyze the provided data and determine the most appropriate chart types based on technical feasibility and visualization best practices.

USER QUERY: {user_query}
AI RESPONSE: {response_text}

DATASET ANALYSIS:
{json.dumps(dataset_info, indent=2)}

COLUMN SCHEMA:
{json.dumps(schema, indent=2)}

SAMPLE DATA STRUCTURE:
{json.dumps(head_sample, indent=2)}

AVAILABLE CHART COMPONENTS WITH CODE REQUIREMENTS:
{json.dumps(chart_options, indent=2)}

TECHNICAL ANALYSIS REQUIRED:
1. **Data Structure Compatibility**: Check if current data can be transformed to required format
2. **Column Type Verification**: Ensure sufficient categorical/numeric/date columns exist
3. **Data Volume Appropriateness**: Match data size to chart type capabilities
4. **User Intent Alignment**: Select charts that answer the user's question effectively
5. **Visualization Clarity**: Avoid charts that would be cluttered or unreadable

CURRENT DATA CAPABILITIES:
- Total Entries: {total_entries}
- Total Columns: {total_columns}
- Categorical Columns: {dataset_info['categorical_columns']} 
- Numeric Columns: {dataset_info['numeric_columns']}
- Data Size Category: {dataset_info['data_size']}
- Available Columns: {[f"{col['name']} ({col['type']})" for col in schema]}

SELECTION LOGIC:
- Index 0: Requires 1 categorical + 1 numeric (good for: rankings, comparisons)
- Index 1: Requires 1 categorical + 2+ numeric (good for: composition, multi-metric)
- Index 2: Requires 1 categorical + 1 numeric, max 8 categories (good for: proportions)
- Index 3: Requires sequential + 1 numeric (good for: trends, patterns)
- Index 4: Requires sequential + 2+ numeric (good for: trend comparison)

RESPONSE REQUIREMENTS:
Return ONLY valid JSON with no additional text. For each recommended graph, provide all necessary parameters:

{{
  "number_of_appropriate_graphs": <integer>,
  "graph_indices": [<array of valid indices 0-4>],
  "graph_configs": [
    {{
      "index": <chart_index>,
      "title": "<descriptive chart title>",
      "description": "<brief explanation of what this chart shows>",
      "x_axis_label": "<x-axis label>",
      "y_axis_label": "<y-axis label>",
      "data_keys": {{
        "primary_key": "<column name for primary axis/categories>",
        "value_keys": ["<column names for numeric values>"],
        "color_key": "<column name for colors/grouping if needed>"
      }},
      "chart_specific": {{
        "code_snippets": "Use these React component patterns for implementation",
        "bar_chart_active": "BarChart with Bar dataKey=value, activeIndex for highlighting, Rectangle activeBar component",
        "bar_chart_stacked": "BarChart with multiple Bar components, stackId=a, ChartLegend for series identification", 
        "pie_chart_donut": "PieChart with Pie innerRadius=60, Label component for center text display",
        "line_chart_dots": "LineChart with Line type=natural, custom Dot components with individual colors",
        "line_chart_multiple": "LineChart with multiple Line components, different stroke colors per series"
      }}
    }}
  ]
}}

REACT COMPONENT CODE REFERENCES:
For implementation guidance, use these component patterns:

BAR CHART ACTIVE (Index 0):
<ChartContainer config={{chartConfig}}>
  <BarChart accessibilityLayer data={{chartData}}>
    <CartesianGrid vertical={{false}} />
    <XAxis dataKey="category" tickLine={{false}} axisLine={{false}} />
    <ChartTooltip cursor={{false}} content={{<ChartTooltipContent hideLabel />}} />
    <Bar dataKey="value" strokeWidth={{2}} radius={{8}} activeIndex={{2}} />
  </BarChart>
</ChartContainer>

BAR CHART STACKED (Index 1):
<ChartContainer config={{chartConfig}}>
  <BarChart accessibilityLayer data={{chartData}}>
    <CartesianGrid vertical={{false}} />
    <XAxis dataKey="category" />
    <ChartLegend content={{<ChartLegendContent />}} />
    <Bar dataKey="series1" stackId="a" fill="var(--color-series1)" />
    <Bar dataKey="series2" stackId="a" fill="var(--color-series2)" />
  </BarChart>
</ChartContainer>

PIE CHART DONUT (Index 2):
<ChartContainer config={{chartConfig}} className="mx-auto aspect-square max-h-[250px]">
  <PieChart>
    <ChartTooltip cursor={{false}} content={{<ChartTooltipContent hideLabel />}} />
    <Pie data={{chartData}} dataKey="value" nameKey="category" innerRadius={{60}} strokeWidth={{2}}>
      <Label content={{centerLabelFunction}} />
    </Pie>
  </PieChart>
</ChartContainer>

LINE CHART DOTS (Index 3):
<ChartContainer config={{chartConfig}}>
  <LineChart accessibilityLayer data={{chartData}} margin={{{{top: 24, left: 24, right: 24}}}}>
    <CartesianGrid vertical={{false}} />
    <ChartTooltip cursor={{false}} />
    <Line dataKey="value" type="natural" strokeWidth={{2}} dot={{customDotFunction}} />
  </LineChart>
</ChartContainer>

LINE CHART MULTIPLE (Index 4):
<ChartContainer config={{chartConfig}}>
  <LineChart accessibilityLayer data={{chartData}}>
    <CartesianGrid vertical={{false}} />
    <XAxis dataKey="category" />
    <ChartTooltip cursor={{false}} />
    <Line dataKey="series1" type="monotone" stroke="var(--color-series1)" strokeWidth={{2}} dot={{false}} />
    <Line dataKey="series2" type="monotone" stroke="var(--color-series2)" strokeWidth={{2}} dot={{false}} />
  </LineChart>
</ChartContainer>

PARAMETER GENERATION GUIDELINES:
1. **Titles**: Create descriptive, context-aware titles based on user query and data
2. **Axis Labels**: Clear, professional labels that describe what's being measured
3. **Data Keys**: Map actual column names from the schema to chart requirements
4. **Colors**: Suggest appropriate color schemes or highlight important segments
5. **Chart-Specific Config**: Include any special settings needed for optimal visualization

CONSTRAINTS:
- Only suggest charts where data requirements are technically feasible
- Prioritize charts that effectively communicate the user's query intent
- Consider data volume for optimal user experience
- Maximum 3 charts to avoid overwhelming the user
- Minimum 1 chart if any are feasible
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        
        # Validate and sanitize response
        if not isinstance(result.get('number_of_appropriate_graphs'), int):
            raise ValueError("Invalid number_of_appropriate_graphs")
        if not isinstance(result.get('graph_indices'), list):
            raise ValueError("Invalid graph_indices")
        if not isinstance(result.get('graph_configs'), list):
            raise ValueError("Invalid graph_configs")
            
        valid_indices = [idx for idx in result['graph_indices'] if isinstance(idx, int) and 0 <= idx <= 4]
        result['graph_indices'] = valid_indices
        result['number_of_appropriate_graphs'] = len(valid_indices)
        result['data']= data  # Include original data for reference
        
        # Filter graph_configs to match valid indices
        valid_configs = [config for config in result.get('graph_configs', []) 
                        if config.get('index') in valid_indices]
        result['graph_configs'] = valid_configs
        
        return result
        
    except Exception as e:
        logger.error(f"Error in decide_graph_from_string: {e}")
        # Enhanced fallback logic
        numeric_cols = dataset_info['numeric_columns']
        categorical_cols = dataset_info['categorical_columns']
        data_size = dataset_info['data_size']
        
        # Enhanced fallback with basic configurations
        def create_fallback_config(index, title, categorical_col, numeric_cols):
            config = {
                "index": index,
                "title": title,
                "description": f"Fallback chart showing {title.lower()}",
                "x_axis_label": categorical_col if categorical_col else "Categories",
                "y_axis_label": numeric_cols[0] if numeric_cols else "Values",
                "data_keys": {
                    "primary_key": categorical_col if categorical_col else schema[0]['name'],
                    "value_keys": numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                    "color_key": None
                },
                "chart_specific": {}
            }
            return config
        
        # Get available columns
        cat_cols = [col['name'] for col in schema if col['type'] == 'categorical']
        num_cols = [col['name'] for col in schema if col['type'] == 'numeric']
        
        # Smart fallback based on data characteristics
        if numeric_cols >= 2 and categorical_cols >= 1:
            if data_size == "small":
                configs = [
                    create_fallback_config(1, "Stacked Comparison", cat_cols[0] if cat_cols else None, num_cols),
                    create_fallback_config(2, "Proportional Distribution", cat_cols[0] if cat_cols else None, num_cols[:1])
                ]
                return {"number_of_appropriate_graphs": 2, "graph_indices": [1, 2], "graph_configs": configs}
            else:
                configs = [
                    create_fallback_config(1, "Multi-Series Analysis", cat_cols[0] if cat_cols else None, num_cols),
                    create_fallback_config(4, "Trend Comparison", cat_cols[0] if cat_cols else None, num_cols)
                ]
                return {"number_of_appropriate_graphs": 2, "graph_indices": [1, 4], "graph_configs": configs}
        elif numeric_cols >= 1 and categorical_cols >= 1:
            if data_size == "small" and categorical_cols <= 8:
                configs = [
                    create_fallback_config(0, "Categorical Comparison", cat_cols[0] if cat_cols else None, num_cols),
                    create_fallback_config(2, "Distribution Analysis", cat_cols[0] if cat_cols else None, num_cols[:1])
                ]
                return {"number_of_appropriate_graphs": 2, "graph_indices": [0, 2], "graph_configs": configs}
            else:
                config = create_fallback_config(0, "Data Overview", cat_cols[0] if cat_cols else None, num_cols)
                return {"number_of_appropriate_graphs": 1, "graph_indices": [0], "graph_configs": [config]}
        else:
            config = create_fallback_config(0, "Basic Chart", schema[0]['name'] if schema else "Category", 
                                          [schema[1]['name']] if len(schema) > 1 else ["Value"])
            return {"number_of_appropriate_graphs": 1, "graph_indices": [0], "graph_configs": [config]}

def clean_md(text: str) -> str:
    # Remove headings (#), emphasis (*, _, **, __), inline code (`), blockquotes (>), lists (-, +)
    text = re.sub(r'[#*_`>\-\+]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_forecast_data(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Cleans a DataFrame by:
      1. Removing non-numeric columns.
      2. Replacing negative values with NaN.
      3. Filling NaN values with the average of up to `window` neighbors
         above and below in the same column.
      4. If no valid neighbors exist, uses the column's overall mean.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        window (int): Number of neighbors to consider above and below.
    
    Returns:
        pd.DataFrame: Cleaned numeric-only DataFrame.
    """
    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    
    # Convert negatives to NaN
    df_numeric[df_numeric < 0] = np.nan
    
    for col in df_numeric.columns:
        col_mean = df_numeric[col].mean(skipna=True)
        
        for idx in df_numeric.index[df_numeric[col].isna()]:
            start = max(0, idx - window)
            end = min(len(df_numeric), idx + window + 1)
            
            neighbors = df_numeric[col].iloc[start:end].dropna()
            
            if len(neighbors) > 0:
                df_numeric.at[idx, col] = neighbors.mean()
            else:
                df_numeric.at[idx, col] = col_mean
                
    return df_numeric

def create_sliding_windows(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def forecast_data(df: pd.DataFrame, cols:List, predict_all_columns=True, forecast_years=5, seq_len=3, epochs=300, lr=0.01):
    if cols==[]:
        cols = df.columns.tolist()
    df = df[cols]
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[0] <= seq_len:
        raise ValueError(f"Need more than {seq_len} rows for the sliding window approach.")
    
    results = pd.DataFrame(index=range(forecast_years), columns=df_numeric.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))  # strictly non-negative scaling
    scaled_data = scaler.fit_transform(df_numeric.values)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if predict_all_columns:
        X, y = create_sliding_windows(scaled_data, seq_len)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        model = GRUPredictor(input_size=df_numeric.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        last_input = torch.tensor(scaled_data[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        for year in range(forecast_years):
            with torch.no_grad():
                next_pred = model(last_input)
            results.iloc[year] = scaler.inverse_transform(next_pred.cpu().numpy())
            last_input = torch.cat([last_input[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
    
    else:
        for i, col in enumerate(df_numeric.columns):
            col_data = scaled_data[:, i:i+1]
            X, y = create_sliding_windows(col_data, seq_len)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
            
            model = GRUPredictor(input_size=1).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()
            
            last_input = torch.tensor(col_data[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
            for year in range(forecast_years):
                with torch.no_grad():
                    next_pred = model(last_input)
                full_pred = np.zeros((1, df_numeric.shape[1]))
                full_pred[0, i] = next_pred.cpu().numpy()[0, 0]
                results.iloc[year, i] = scaler.inverse_transform(full_pred)[0, i]
                last_input = torch.cat([last_input[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
    
    results = results.to_csv(index=False)
    return results


def eda_analysis(df: pd.DataFrame, user_query: str) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    """
    Generate and execute EDA code using Gemini with Plotly visualizations.
    
    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze
        user_query (str): User's analysis request
        
    Returns:
        dict: Analysis results with plotly figures as JSON
    """
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    
    # Get DataFrame info for context
    df_info = f"""
DataFrame Info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data Types: {dict(df.dtypes)}
- Numeric Columns: {list(df.select_dtypes(include=[np.number]).columns)}
- Categorical Columns: {list(df.select_dtypes(include=['object', 'category']).columns)}
- Missing Values: {dict(df.isnull().sum())}
"""
    
    # Create standardized prompt
    prompt = f"""
{df_info}

USER REQUEST: {user_query}

Generate Python code that creates a function called `analyze_data(df)` that:

1. Takes a pandas DataFrame 'df' as input
2. Performs the requested analysis
3. Returns a dictionary called 'result' with analysis findings
4. Creates visualizations using Plotly (plotly.express as px, plotly.graph_objects as go)
5. Store Plotly figures in the result dictionary as 'figures' key (list of figure objects)

IMPORTANT RULES:
- Only generate the function code, no explanations
- Use available libraries: pandas (pd), plotly.express (px), plotly.graph_objects (go), plotly.figure_factory (ff), numpy (np)
- Handle edge cases (empty data, missing values, etc.)
- Store all computed results in the 'result' dictionary
- Store all plotly figures in result['figures'] as a list
- Use descriptive keys in the result dictionary
- Do NOT call fig.show() - just create and store the figures
- Choose the best variant of plots for each type of data and analysis. Always include a correlation matrix and a line graph if applicable

EXAMPLE STRUCTURE:
def analyze_data(df):
    result = {{'figures': []}}
    
    # Your analysis code here
    fig = px.histogram(df, x='column_name')
    result['figures'].append(fig)
    
    # More analysis...
    result['analysis_summary'] = {{...}}
    
    return result

Generate only executable Python code:
"""
    
    try:
        # Generate code with Gemini
        response = model.generate_content(prompt)
        
        # Extract code from response
        code = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, "text"):
                        text = part.text.strip()
                        # Clean up code blocks
                        if "```python" in text:
                            code = text.split("```python")[1].split("```")[0].strip()
                        elif "```" in text:
                            code = text.split("```")[1].split("```")[0].strip()
                        else:
                            code = text
                        break
        
        if not code:
            raise ValueError("No code generated by Gemini")
        
        # Execute the generated code
        exec_globals = {
            'pd': pd, 'px': px, 'go': go, 'ff': ff, 'np': np,
            'df': df
        }
        exec_locals = {}
        
        # Execute the generated code
        exec(code, exec_globals, exec_locals)
        
        # Call the analysis function
        if 'analyze_data' in exec_locals:
            result = exec_locals['analyze_data'](df)
            
            # Convert Plotly figures to JSON for frontend - FIXED VERSION
            if isinstance(result, dict) and 'figures' in result:
                plotly_json_figures = []
                for fig in result['figures']:
                    try:
                        # Convert each figure to JSON-serializable format
                        if hasattr(fig, 'to_dict'):
                            fig_dict = fig.to_dict()
                            plotly_json_figures.append(fig_dict)
                        elif hasattr(fig, 'to_json'):
                            fig_json = json.loads(fig.to_json())
                            plotly_json_figures.append(fig_json)
                        else:
                            # Fallback: try to serialize as dict
                            plotly_json_figures.append(dict(fig))
                    except Exception as fig_error:
                        logger.warning(f"Could not serialize figure: {fig_error}")
                        # Add a placeholder for failed figures
                        plotly_json_figures.append({
                            "error": f"Figure serialization failed: {str(fig_error)}",
                            "type": "error"
                        })
                
                # Replace the figures with JSON-serializable versions
                result['plotly_json'] = plotly_json_figures
                # Remove original figures to avoid serialization issues
                del result['figures']
            
            # Ensure all values in result are JSON serializable
            result = convert_numpy_types(result)
            
            return result if isinstance(result, dict) else {'analysis': str(result)}
        else:
            return {'error': 'analyze_data function not found in generated code'}
            
    except Exception as e:
        logger.error(f"EDA analysis error: {e}")
        return {
            'error': f'Analysis failed: {str(e)}',
            'fallback_info': {
                'dataframe_shape': list(df.shape),
                'columns': list(df.columns),
                'numeric_summary': convert_numpy_types(
                    df.select_dtypes(include=[np.number]).describe().to_dict() 
                    if not df.select_dtypes(include=[np.number]).empty else {}
                )
            }
        }

def display_plotly_figures(result: dict):
    """
    Helper function to display Plotly figures locally.
    
    Args:
        result (dict): Result from gemini_eda_analysis containing plotly_figures
    """
    if 'plotly_figures' in result:
        for i, fig in enumerate(result['plotly_figures']):
            print(f"Displaying Figure {i+1}")
            fig.show()
    else:
        print("No figures found in result")

def get_plotly_json_for_frontend(result: dict) -> list:
    """
    Extract Plotly JSON data for frontend rendering.
    
    Args:
        result (dict): Result from gemini_eda_analysis
        
    Returns:
        list: List of Plotly JSON objects for frontend
    """
    return result.get('plotly_json', [])