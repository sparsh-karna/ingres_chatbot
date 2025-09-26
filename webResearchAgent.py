"""
Web Research Agent - Simulates web research for policy analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Web Research Agent")

# Add CORS middleware: Cross-Origin Resource Sharing
# CORS is a security mechanism built into browsers. It determines whether a web page from one domain can make requests to another domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ResearchRequest(BaseModel): # defining the expected structure of request body
    search_terms: List[str]
    focus_areas: List[str] = []
    context: Optional[str] = None

class ResearchResponse(BaseModel):  # defining the response body your API will return.
    findings: List[str]
    sources: List[str]
    summary: str
    confidence: str

def create_research_prompt(search_terms: List[str], focus_areas: List[str], context: str = None) -> str:
    
    prompt = f"""You are a policy research expert conducting web research in India. Based on the search terms and focus areas provided, generate realistic research findings that would be found through web searches.

SEARCH TERMS: {', '.join(search_terms)}
FOCUS AREAS: {', '.join(focus_areas)}
{f"CONTEXT: {context}" if context else ""}

Generate realistic research findings as if you conducted actual web searches. Include:
1. Current government policies and initiatives
2. Latest news and developments
3. Best practices from other regions
4. Expert opinions and recommendations
5. Statistical data and trends

Return your response as JSON with this structure:
{{
    "findings": [
        "Finding 1: Specific policy or initiative",
        "Finding 2: Recent development or news",
        "Finding 3: Best practice or case study",
        "Finding 4: Expert recommendation",
        "Finding 5: Statistical insight"
    ],
    "sources": [
        "Government agency or department",
        "News organization",
        "Research institution",
        "Policy think tank",
        "Academic publication"
    ],
    "summary": "Brief summary of key research findings",
    "confidence": "High|Medium|Low"
}}

Make the findings specific, realistic, and relevant to the search terms. Focus on actionable insights for policy makers."""

    return prompt

def parse_research_response(response_text: str) -> Dict[str, Any]:
    """Parse the research response"""
    try:
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
            
    except json.JSONDecodeError:
        # Fallback research results
        return {
            "findings": [
                "Current government initiatives identified",
                "Recent policy developments found",
                "Best practices from similar regions documented",
                "Expert recommendations compiled",
                "Statistical trends analyzed"
            ],
            "sources": [
                "Government agencies",
                "Policy research institutions",
                "News organizations",
                "Academic publications",
                "Think tanks"
            ],
            "summary": "Research completed on specified topics with relevant findings identified",
            "confidence": "Medium"
        }

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """Conduct web research simulation"""
    
    try:
        
        # Create research prompt
        prompt = create_research_prompt(
            request.search_terms,
            request.focus_areas,
            request.context
        )
        
        # Call Gemini API for research simulation
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from AI model")
        
        # Parse response
        research_data = parse_research_response(response.text)
        
        return ResearchResponse(
            findings=research_data.get("findings", []),
            sources=research_data.get("sources", []),
            summary=research_data.get("summary", "Research completed"),
            confidence=research_data.get("confidence", "Medium")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Web Research Agent API is running"}

@app.get("/test-research")
async def test_research():
    """Test research with sample query"""
    test_request = ResearchRequest(
        search_terms=["groundwater management policy", "northeastern states"],
        focus_areas=["government initiatives", "regulations", "best practices"],
        context="Policy analysis for water resource management"
    )
    
    return await conduct_research(test_request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
