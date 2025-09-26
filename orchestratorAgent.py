"""
Orchestrator Agent - Main agent that breaks down complex policy queries into subtasks
Handles complex queries like "Prepare a Plan to increase groundwater levels in the northeastern states"
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Orchestrator Agent", description="Break down complex policy queries into executable tasks")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Task(BaseModel):
    id: str
    type: str  # "data_analysis", "web_research", "visualization", "calculation"
    title: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = []  # Task IDs this task depends on
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None

class OrchestratorRequest(BaseModel):
    query: str
    csv_files: List[str] = ["2024-2025.csv"]
    index_files: List[str] = ["index_2024-2025.txt"]
    context: Optional[str] = None  # Additional context about the policy domain

class OrchestratorResponse(BaseModel):
    plan_id: str
    original_query: str
    tasks: List[Task]
    estimated_duration: str
    complexity_level: str  # "simple", "moderate", "complex"
    reasoning: str

def create_orchestrator_prompt(query: str, csv_files: List[str], index_files: List[str], context: str = None) -> str:
    """Create prompt for the orchestrator to break down complex queries"""

    # Read the index file to understand data structure
    index_content = ""
    if index_files:
        try:
            with open(index_files[0], 'r') as f:
                index_content = f.read()
        except:
            index_content = "Index file not available"

    prompt = f"""You are an expert policy analyst and data scientist orchestrator. Your job is to break down complex policy queries into specific, executable tasks.

AVAILABLE DATA SOURCES:
- CSV Files: {', '.join(csv_files)}
- Index Files: {', '.join(index_files)}
- Web Research: Available for current news, policies, and external data
- Visualization Tools: Plotly, Matplotlib for charts and graphs

CSV DATA STRUCTURE:
{index_content[:2000]}...

USER QUERY: {query}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

IMPORTANT: Based on the CSV data structure above, you can see the exact columns available:
- STATE: Contains Indian states (UTTAR PRADESH, MADHYA PRADESH, BIHAR, TAMILNADU, MAHARASHTRA, GUJARAT, etc.)
- DISTRICT: District-level data
- Total Rainfall (mm): Rainfall data in millimeters
- Annual Ground water Recharge (ham): Annual groundwater recharge in hectare-meters
- Total Ground Water Availability in the area (ham) - Fresh: Total fresh groundwater availability
- Stage of Ground Water Extraction (%): Percentage of groundwater extraction
- Net Annual Ground Water Availability for Future Use (ham): Available groundwater for future use
- Ground Water Extraction for all uses (ham) - Total: Total groundwater extraction
- Ground Water Extraction for all uses (ham) - Domestic: Domestic water extraction
- Ground Water Extraction for all uses (ham) - Industrial: Industrial water extraction
- Ground Water Extraction for all uses (ham) - Irrigation: Irrigation water extraction

For northeastern Indian states, focus on: ASSAM, ARUNACHAL PRADESH, MANIPUR, MEGHALAYA, MIZORAM, NAGALAND, TRIPURA, SIKKIM

CRITICAL: Always use the EXACT column names as shown above. Do not use generic terms like 'groundwater_level' - use the specific column names like 'Annual Ground water Recharge (ham)' or 'Total Ground Water Availability in the area (ham) - Fresh'.

Break this query into specific tasks that can be executed by specialized agents. Each task should be:
1. Specific and actionable with exact column names
2. Have clear parameters referencing actual CSV columns
3. Specify dependencies on other tasks
4. Include expected output type
5. Generate detailed visualizations and analysis

TASK TYPES AVAILABLE:
- data_analysis: Query and analyze CSV data using exact column names
- web_research: Search for current news, policies, government initiatives
- visualization: Create detailed charts, graphs, maps with specific data
- calculation: Perform statistical calculations, projections
- synthesis: Combine multiple data sources

Return your response as JSON with this structure:
{{
    "reasoning": "Explanation of how you broke down the query",
    "complexity_level": "simple|moderate|complex",
    "estimated_duration": "X minutes",
    "tasks": [
        {{
            "type": "data_analysis",
            "title": "Brief task title",
            "description": "Detailed description of what to do",
            "parameters": {{
                "query": "Specific data query using EXACT column names from the CSV schema above",
                "csv_file": "2024-2025.csv",
                "analysis_type": "statistical|trend|comparison|correlation"
            }},
            "dependencies": []
        }},
        {{
            "type": "web_research",
            "title": "Research current policies",
            "description": "Search for recent government initiatives",
            "parameters": {{
                "search_terms": ["groundwater policy", "northeastern states"],
                "focus_areas": ["government initiatives", "recent news"]
            }},
            "dependencies": []
        }},
        {{
            "type": "visualization",
            "title": "Create visualization",
            "description": "Generate charts showing trends",
            "parameters": {{
                "chart_type": "bar|line|map|heatmap",
                "data_source": "task_id_reference"
            }},
            "dependencies": ["previous_task_id"]
        }}
    ]
}}

IMPORTANT INSTRUCTIONS:
- For policy queries, always include web research for current initiatives
- For regional queries (like "northeastern states"), include geographical analysis of Indian states
- For planning queries, include trend analysis and projections
- Create visualizations to support findings
- End with a synthesis task that combines all findings
- ALWAYS use the EXACT column names from the CSV schema (e.g., 'Annual Ground water Recharge (ham)', not 'groundwater_level')
- For groundwater analysis, use columns like 'Total Ground Water Availability in the area (ham) - Fresh'
- For extraction analysis, use 'Stage of Ground Water Extraction (%)'
- For rainfall analysis, use 'Total Rainfall (mm)'
- Remember this is Indian groundwater data, not US data

EXAMPLE CORRECT QUERY: "Analyze df['Annual Ground water Recharge (ham)'] for states in ['ASSAM', 'ARUNACHAL PRADESH', 'MANIPUR', 'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'TRIPURA', 'SIKKIM']"

Generate a comprehensive task breakdown now:"""

    return prompt

def parse_orchestrator_response(response_text: str) -> Dict[str, Any]:
    """Parse the orchestrator response and extract task breakdown"""
    try:
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
            
    except json.JSONDecodeError as e:
        # Fallback: create a simple task breakdown
        return {
            "reasoning": "Fallback task breakdown due to parsing error",
            "complexity_level": "moderate",
            "estimated_duration": "15-20 minutes",
            "tasks": [
                {
                    "type": "data_analysis",
                    "title": "Analyze available data",
                    "description": f"Analyze data related to: {response_text[:100]}...",
                    "parameters": {
                        "query": "Analyze the data",
                        "csv_file": "2024-2025.csv",
                        "analysis_type": "statistical"
                    },
                    "dependencies": []
                }
            ]
        }

@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate_query(request: OrchestratorRequest):
    """Break down complex query into executable tasks"""
    
    try:
        # Create orchestrator prompt
        prompt = create_orchestrator_prompt(
            request.query, 
            request.csv_files, 
            request.index_files, 
            request.context
        )
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from AI model")
        
        # Parse response
        task_breakdown = parse_orchestrator_response(response.text)
        
        # Generate unique IDs for tasks and plan
        plan_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Create Task objects with unique IDs
        tasks = []
        for i, task_data in enumerate(task_breakdown.get("tasks", [])):
            task = Task(
                id=str(uuid.uuid4()),
                type=task_data.get("type", "data_analysis"),
                title=task_data.get("title", f"Task {i+1}"),
                description=task_data.get("description", ""),
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", []),
                created_at=current_time
            )
            tasks.append(task)
        
        return OrchestratorResponse(
            plan_id=plan_id,
            original_query=request.query,
            tasks=tasks,
            estimated_duration=task_breakdown.get("estimated_duration", "10-15 minutes"),
            complexity_level=task_breakdown.get("complexity_level", "moderate"),
            reasoning=task_breakdown.get("reasoning", "Task breakdown completed")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Orchestrator Agent API is running"}

@app.get("/test-orchestration")
async def test_orchestration():
    """Test the orchestration with a sample complex query"""
    test_request = OrchestratorRequest(
        query="Prepare a Plan to increase groundwater levels in the northeastern states",
        csv_files=["2024-2025.csv"],
        index_files=["index_2024-2025.txt"],
        context="Policy analysis for water resource management"
    )
    
    return await orchestrate_query(test_request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)


