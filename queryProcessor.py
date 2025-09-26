from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
import uvicorn
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="CSV Query Processor", description="Generate Python code from natural language queries using Gemini AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

class QueryRequest(BaseModel):
    query: str
    index_file: str = "index_2024-2025.txt"
    csv_file: str = "2024-2025.csv"

class CodeResponse(BaseModel):
    code: str
    reasoning: str
    instructions: str
    query: str
    index_file: str
    csv_file: str

def get_available_index_files() -> List[str]:
    """Get list of available index files"""
    index_files = glob.glob("index_*.txt")
    return sorted(index_files)

def get_available_csv_files() -> List[str]:
    """Get list of available CSV files"""
    csv_files = glob.glob("*.csv")
    return sorted(csv_files)

def load_index_file(filename: str) -> str:
    """Load and return content of index file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Index file '{filename}' not found")
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def create_prompt(query: str, index_content: str, csv_filename: str) -> str:
    """Create optimized prompt for Gemini with Plotly visualization support"""

    # Detect if query is asking for visualization
    visual_keywords = ['plot', 'chart', 'graph', 'visualize', 'show', 'display', 'bar chart', 'line chart', 'scatter', 'histogram', 'pie chart', 'heatmap', 'trend', 'distribution', 'comparison']
    is_visual_query = any(keyword in query.lower() for keyword in visual_keywords)

    if is_visual_query:
        prompt = f"""You are an expert data analyst and Python programmer. Your task is to generate Python code that answers the user's query about a CSV dataset with INTERACTIVE VISUALIZATION.

DATASET INFORMATION:
{index_content}

USER QUERY: {query}

VISUALIZATION INSTRUCTIONS:
1. Generate clean, efficient Python code using pandas for data processing
2. Use EXACTLY the column names as shown in the COLUMN SCHEMA section
3. For VISUALIZATION queries, use Plotly Express (px) or Plotly Graph Objects (go) for interactive charts
4. Import plotly.express as px and/or plotly.graph_objects as go
5. Create interactive plots with proper titles, labels, and hover information
6. Use fig.show() to display the plot (this will be captured for web display)
7. Handle missing data appropriately based on the null percentages shown
8. Include proper error handling and data validation
9. Add clear comments explaining each step
10. Use the sample data to understand value formats
11. Load the CSV file using: df = pd.read_csv('{csv_filename}')
12. Focus on answering the specific question asked

PLOTLY EXAMPLES:
- Bar chart: px.bar(df, x='column1', y='column2', title='Title', hover_data=['additional_col'])
- Line chart: px.line(df, x='column1', y='column2', title='Title', markers=True)
- Scatter plot: px.scatter(df, x='column1', y='column2', title='Title', color='category_col')
- Histogram: px.histogram(df, x='column1', title='Title', nbins=30)
- Pie chart: px.pie(df, values='column1', names='column2', title='Title')
- Box plot: px.box(df, x='category', y='value', title='Title')
- Heatmap: px.imshow(correlation_matrix, title='Correlation Heatmap')"""
    else:
        prompt = f"""You are an expert data analyst and Python programmer. Your task is to generate Python code that answers the user's query about a CSV dataset.

DATASET INFORMATION:
{index_content}

USER QUERY: {query}

INSTRUCTIONS:
1. Generate clean, efficient Python code using pandas
2. Use EXACTLY the column names as shown in the COLUMN SCHEMA section
3. Handle missing data appropriately based on the null percentages shown
4. Include proper error handling and data validation
5. Add clear comments explaining each step
6. Use the sample data to understand value formats
7. Load the CSV file using: df = pd.read_csv('{csv_filename}')
8. Focus on answering the specific question asked
9. If visualization would help answer the query, use plotly.express for interactive charts

RESPONSE FORMAT:
Please provide your response in the following JSON format:

{{
    "reasoning": "Explain your approach, what columns you'll use, any data considerations, and step-by-step logic",
    "code": "Complete Python code that answers the query",
    "instructions": "Brief instructions on how to run this code and what to expect"
}}

IMPORTANT:
- Only return valid JSON
- Use proper escaping for quotes in the JSON
- Make sure the code is complete and runnable
- Consider data types and missing values from the schema
- Be specific about which columns and operations you're using"""

    return prompt

def parse_gemini_response(response_text: str) -> Dict[str, str]:
    """Parse Gemini response and extract JSON"""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            return {
                "reasoning": parsed.get("reasoning", ""),
                "code": parsed.get("code", ""),
                "instructions": parsed.get("instructions", "")
            }
        else:
            # Fallback: try to parse the entire response as JSON
            parsed = json.loads(response_text)
            return {
                "reasoning": parsed.get("reasoning", ""),
                "code": parsed.get("code", ""),
                "instructions": parsed.get("instructions", "")
            }
    except json.JSONDecodeError:
        # Fallback: extract code blocks and reasoning manually
        code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        code = code_match.group(1) if code_match else response_text
        
        return {
            "reasoning": "Generated code based on query analysis",
            "code": code,
            "instructions": "Run this code after loading your CSV file with pandas"
        }

@app.get("/")
async def root():
    return {"message": "CSV Query Processor API is running"}

@app.get("/index-files")
async def list_index_files():
    """Get list of available index files"""
    try:
        files = get_available_index_files()
        return {"index_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing index files: {str(e)}")

@app.get("/csv-files")
async def list_csv_files():
    """Get list of available CSV files"""
    try:
        files = get_available_csv_files()
        return {"csv_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing CSV files: {str(e)}")

@app.post("/generate-code", response_model=CodeResponse)
async def generate_code(request: QueryRequest):
    """Generate Python code from natural language query"""
    
    try:
        # Validate inputs
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Load index file
        try:
            index_content = load_index_file(request.index_file)
        except FileNotFoundError:
            available_files = get_available_index_files()
            raise HTTPException(
                status_code=404, 
                detail=f"Index file '{request.index_file}' not found. Available files: {available_files}"
            )
        
        # Create prompt
        prompt = create_prompt(request.query, index_content, request.csv_file)
        
        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            
            if not response.text:
                raise HTTPException(status_code=500, detail="Empty response from Gemini API")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
        
        # Parse response
        try:
            parsed_response = parse_gemini_response(response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing Gemini response: {str(e)}")
        
        # Validate parsed response
        if not parsed_response.get("code"):
            raise HTTPException(status_code=500, detail="No code generated in response")
        
        return CodeResponse(
            code=parsed_response["code"],
            reasoning=parsed_response["reasoning"],
            instructions=parsed_response["instructions"],
            query=request.query,
            index_file=request.index_file,
            csv_file=request.csv_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/test-gemini")
async def test_gemini():
    """Test Gemini API connection"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say hello and confirm you're working!")
        return {"status": "success", "response": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
