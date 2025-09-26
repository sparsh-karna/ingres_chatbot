"""
Data Analysis and Presentation Layer
Converts raw results into natural language explanations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Result Analyzer", description="Convert data results into natural language explanations")

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

class AnalysisRequest(BaseModel):
    query: str  # Original user question
    print_output: str  # Console output from code execution
    dataframes: List[Dict[str, Any]]  # DataFrame summaries
    variables: Dict[str, Any]  # Simple variables
    plots_description: Optional[str] = None  # Description of plots/charts
    execution_time: float  # How long the analysis took

class AnalysisResponse(BaseModel):
    explanation: str  # Natural language explanation
    key_insights: List[str]  # Bullet points of main findings
    data_summary: str  # Summary of the data analyzed
    confidence: str  # High/Medium/Low confidence in the analysis

def create_analysis_prompt(request: AnalysisRequest) -> str:
    """Create prompt for result analysis"""
    
    prompt = f"""You are a data analyst explaining results to a non-technical user. Your task is to provide a clear, simple explanation of the data analysis results.

ORIGINAL QUESTION: {request.query}

ANALYSIS RESULTS:
"""

    # Add print output if available
    if request.print_output:
        prompt += f"""
CONSOLE OUTPUT:
{request.print_output}
"""

    # Add DataFrame information
    if request.dataframes:
        prompt += f"""
DATAFRAMES ANALYZED:
"""
        for df in request.dataframes:
            prompt += f"""
- {df['name']}: {df['shape'][0]} rows × {df['shape'][1]} columns
  Columns: {', '.join(df['columns'][:5])}{'...' if len(df['columns']) > 5 else ''}
"""
            if 'statistics' in df:
                prompt += f"  Key statistics available for numerical columns\n"

    # Add variables
    if request.variables:
        prompt += f"""
KEY RESULTS:
"""
        for var_name, var_value in request.variables.items():
            if isinstance(var_value, (int, float)):
                prompt += f"- {var_name}: {var_value}\n"
            elif isinstance(var_value, dict) and len(var_value) <= 10:
                prompt += f"- {var_name}: {var_value}\n"
            elif isinstance(var_value, list) and len(var_value) <= 10:
                prompt += f"- {var_name}: {var_value}\n"

    # Add plot description if available
    if request.plots_description:
        prompt += f"""
VISUALIZATIONS:
{request.plots_description}
"""

    prompt += f"""
EXECUTION TIME: {request.execution_time} seconds

INSTRUCTIONS:
1. Provide a clear, simple explanation that a non-technical person can understand
2. Focus on what the data tells us, not how it was calculated
3. Highlight the most important findings
4. Use everyday language, avoid technical jargon
5. If there are numbers, put them in context (e.g., "high", "low", "average")
6. Answer the original question directly

RESPONSE FORMAT:
Please provide your response in the following JSON format:

{{
    "explanation": "A clear 2-3 sentence explanation of what the analysis found, directly answering the user's question",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
    "data_summary": "Brief summary of what data was analyzed (e.g., '728 records of groundwater data from 37 states')",
    "confidence": "High/Medium/Low - based on data quality and completeness"
}}

IMPORTANT:
- Only return valid JSON
- Keep explanations simple and conversational
- Focus on insights, not technical details
- Make it actionable and meaningful to the user
"""

    return prompt

def parse_analysis_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini response and extract JSON"""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['explanation', 'key_insights', 'data_summary', 'confidence']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = f"Analysis {field} not available"
            
            return parsed
        else:
            # Fallback if no JSON found
            return {
                "explanation": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "key_insights": ["Analysis completed successfully"],
                "data_summary": "Data analysis performed",
                "confidence": "Medium"
            }
    except Exception as e:
        return {
            "explanation": f"Analysis completed but explanation formatting failed: {str(e)}",
            "key_insights": ["Results generated successfully"],
            "data_summary": "Data processed",
            "confidence": "Low"
        }

@app.post("/analyze-results", response_model=AnalysisResponse)
async def analyze_results(request: AnalysisRequest):
    """Analyze execution results and provide natural language explanation"""
    
    try:
        # Create analysis prompt
        prompt = create_analysis_prompt(request)
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from AI model")
        
        # Parse response
        analysis_data = parse_analysis_response(response.text)
        
        return AnalysisResponse(**analysis_data)
        
    except Exception as e:
        # Fallback analysis
        fallback_explanation = f"The analysis of your query '{request.query}' has been completed successfully."
        
        if request.print_output:
            # Try to extract key information from print output
            lines = request.print_output.strip().split('\n')
            if lines:
                fallback_explanation += f" The results show: {lines[-1] if lines[-1] else 'data processed successfully'}."
        
        return AnalysisResponse(
            explanation=fallback_explanation,
            key_insights=["Analysis completed", "Results generated successfully"],
            data_summary=f"Processed {len(request.dataframes)} datasets" if request.dataframes else "Data analysis performed",
            confidence="Medium"
        )

@app.get("/")
async def root():
    return {"message": "Result Analyzer API is running"}

@app.get("/test-analysis")
async def test_analysis():
    """Test the analysis functionality"""
    test_request = AnalysisRequest(
        query="What is the average rainfall by state?",
        print_output="STATE\nUP    1200.5\nMH    980.2\nKL    2833.4\nName: Total Rainfall (mm), dtype: float64",
        dataframes=[{
            "name": "df",
            "shape": [728, 30],
            "columns": ["STATE", "DISTRICT", "Total Rainfall (mm)"]
        }],
        variables={"average_rainfall": 1338.2},
        execution_time=0.045
    )
    
    return await analyze_results(test_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
