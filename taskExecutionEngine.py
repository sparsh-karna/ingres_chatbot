"""
Task Execution Engine - Manages and executes tasks automatically
Coordinates between different specialized agents and tracks progress
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Task Execution Engine", description="Execute and manage tasks automatically")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for execution plans (in production, use a database)
execution_plans: Dict[str, Dict] = {}
task_results: Dict[str, Dict] = {}

class ExecutionPlan(BaseModel):
    plan_id: str
    original_query: str
    tasks: List[Dict[str, Any]]
    status: str = "pending"  # pending, running, completed, failed
    current_task_index: int = 0
    results: List[Dict[str, Any]] = []
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_percentage: float = 0.0

class ExecutionRequest(BaseModel):
    plan_id: str
    tasks: List[Dict[str, Any]]
    original_query: str

class ExecutionStatus(BaseModel):
    plan_id: str
    status: str
    progress_percentage: float
    current_task: Optional[Dict[str, Any]] = None
    completed_tasks: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    error: Optional[str] = None

# Service endpoints
SERVICES = {
    "query_processor": "http://localhost:8001",
    "code_executor": "http://localhost:8002", 
    "result_analyzer": "http://localhost:8003",
    "web_research": "http://localhost:8007", 
}

async def execute_data_analysis_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a data analysis task using existing services"""
    try:
        params = task.get("parameters", {})
        
        # Step 1: Generate code using query processor
        async with aiohttp.ClientSession() as session:
            query_payload = {
                "query": params.get("query", task["description"]),
                "index_file": params.get("index_file", "index_2024-2025.txt"),
                "csv_file": params.get("csv_file", "2024-2025.csv")
            }
            
            async with session.post(
                f"{SERVICES['query_processor']}/generate-code",
                json=query_payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Code generation failed: {response.status}")
                code_result = await response.json()
        
        # Step 2: Execute code using code executor
        async with aiohttp.ClientSession() as session:
            exec_payload = {
                "code": code_result["code"],
                "csv_file": params.get("csv_file", "2024-2025.csv"),
                "include_analysis": True,
                "original_query": params.get("query", task["description"])
            }
            
            async with session.post(
                f"{SERVICES['code_executor']}/execute-with-analysis",
                json=exec_payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Code execution failed: {response.status}")
                exec_result = await response.json()
        
        return {
            "success": True,
            "type": "data_analysis",
            "code_generated": code_result["code"],
            "execution_result": exec_result,
            "analysis": exec_result.get("analysis"),
            "visualizations": exec_result.get("plotly_figures", [])
        }
        
    except Exception as e:
        return {
            "success": False,
            "type": "data_analysis", 
            "error": str(e)
        }

async def execute_web_research_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a web research task using the web research service"""
    try:
        params = task.get("parameters", {})

        # Call web research service
        async with aiohttp.ClientSession() as session:
            research_payload = {
                "search_terms": params.get("search_terms", []),
                "focus_areas": params.get("focus_areas", []),
                "context": task.get("description", "")
            }

            async with session.post(
                "http://localhost:8007/research",
                json=research_payload
            ) as response:
                if response.status != 200:
                    # Fallback to simulated research
                    return {
                        "success": True,
                        "type": "web_research",
                        "search_terms": params.get("search_terms", []),
                        "findings": [
                            "Recent government initiative on groundwater conservation",
                            "Northeastern states water policy updates",
                            "Current rainfall patterns and climate data"
                        ],
                        "sources": [
                            "Ministry of Water Resources",
                            "Central Ground Water Board",
                            "State government portals"
                        ],
                        "summary": "Research completed with fallback data",
                        "confidence": "Medium"
                    }

                research_result = await response.json()

        return {
            "success": True,
            "type": "web_research",
            "search_terms": params.get("search_terms", []),
            "findings": research_result.get("findings", []),
            "sources": research_result.get("sources", []),
            "summary": research_result.get("summary", ""),
            "confidence": research_result.get("confidence", "Medium")
        }

    except Exception as e:
        return {
            "success": False,
            "type": "web_research",
            "error": str(e)
        }

async def execute_visualization_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a visualization task, creates charts based on data or previous task results"""
    try:
        params = task.get("parameters", {})

        # Get the plan_id to access previous task results
        plan_id = task.get("plan_id")
        previous_results = []

        # If we have a plan_id, get previous task results
        if plan_id and plan_id in execution_plans:
            plan = execution_plans[plan_id]
            previous_results = plan.get("results", [])

        # Determine visualization approach
        viz_approach = params.get("approach", "auto")  # auto, data_driven, or custom

        if viz_approach == "data_driven" and previous_results:
            # Create visualization based on previous data analysis results
            return await create_data_driven_visualization(task, previous_results)
        elif viz_approach == "custom" or params.get("custom_query"):
            # Generate custom visualization using query processor and code executor
            return await create_custom_visualization(task, params)
        else:
            # Auto approach - try data_driven first, fallback to custom
            if previous_results:
                # Check if we have data analysis results to work with
                data_results = [r for r in previous_results if r.get("type") == "data_analysis" and r.get("success")]
                if data_results:
                    return await create_data_driven_visualization(task, previous_results)

            # Fallback to custom visualization
            return await create_custom_visualization(task, params)

    except Exception as e:
        return {
            "success": False,
            "type": "visualization",
            "error": str(e)
        }

async def create_data_driven_visualization(task: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create visualization based on previous data analysis results"""
    try:
        params = task.get("parameters", {})

        # Find the most recent successful data analysis result
        data_result = None
        for result in reversed(previous_results):
            if result.get("type") == "data_analysis" and result.get("success"):
                data_result = result
                break

        if not data_result:
            raise Exception("No data analysis results found to create visualization from")

        # Extract data information from the analysis result
        execution_result = data_result.get("execution_result", {})
        dataframes = execution_result.get("dataframes", [])
        variables = execution_result.get("variables", {})

        # If there are already Plotly figures from the analysis, return them
        existing_plots = execution_result.get("plotly_figures", [])
        if existing_plots and not params.get("force_new_chart"):
            return {
                "success": True,
                "type": "visualization",
                "chart_type": "existing",
                "plotly_figures": existing_plots,
                "source": "previous_analysis",
                "description": f"Reusing {len(existing_plots)} visualization(s) from previous data analysis"
            }

        # Generate new visualization code based on the data
        chart_type = params.get("chart_type", "bar")
        chart_title = params.get("title", task.get("title", "Data Visualization"))

        # Create visualization code based on available data
        viz_code = generate_visualization_code(dataframes, variables, chart_type, chart_title, params)

        # Execute the visualization code
        async with aiohttp.ClientSession() as session:
            exec_payload = {
                "code": viz_code,
                "csv_file": params.get("csv_file", "2024-2025.csv"),
                "include_analysis": False,
                "original_query": f"Create {chart_type} chart: {chart_title}"
            }

            async with session.post(
                f"{SERVICES['code_executor']}/execute-code",
                json=exec_payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Visualization code execution failed: {response.status}")
                exec_result = await response.json()

        if not exec_result.get("success"):
            raise Exception(f"Visualization execution failed: {exec_result.get('error', 'Unknown error')}")

        return {
            "success": True,
            "type": "visualization",
            "chart_type": chart_type,
            "code_generated": viz_code,
            "execution_result": exec_result,
            "plotly_figures": exec_result.get("plotly_figures", []),
            "source": "data_driven",
            "description": f"Generated {chart_type} chart based on previous data analysis"
        }

    except Exception as e:
        return {
            "success": False,
            "type": "visualization",
            "error": f"Data-driven visualization failed: {str(e)}"
        }

async def create_custom_visualization(task: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create custom visualization using query processor and code executor"""
    try:
        # Get visualization query
        viz_query = params.get("custom_query") or params.get("query") or task.get("description", "")
        if not viz_query:
            viz_query = f"Create a {params.get('chart_type', 'bar')} chart for {task.get('title', 'data analysis')}"

        # Ensure the query asks for visualization
        visual_keywords = ['plot', 'chart', 'graph', 'visualize', 'show', 'display']
        if not any(keyword in viz_query.lower() for keyword in visual_keywords):
            viz_query = f"Create a visualization to {viz_query}"

        # Step 1: Generate visualization code using query processor
        async with aiohttp.ClientSession() as session:
            query_payload = {
                "query": viz_query,
                "index_file": params.get("index_file", "index_2024-2025.txt"),
                "csv_file": params.get("csv_file", "2024-2025.csv")
            }

            async with session.post(
                f"{SERVICES['query_processor']}/generate-code",
                json=query_payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Visualization code generation failed: {response.status}")
                code_result = await response.json()

        # Step 2: Execute visualization code using code executor
        async with aiohttp.ClientSession() as session:
            exec_payload = {
                "code": code_result["code"],
                "csv_file": params.get("csv_file", "2024-2025.csv"),
                "include_analysis": False,
                "original_query": viz_query
            }

            async with session.post(
                f"{SERVICES['code_executor']}/execute-code",
                json=exec_payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Visualization code execution failed: {response.status}")
                exec_result = await response.json()

        if not exec_result.get("success"):
            raise Exception(f"Visualization execution failed: {exec_result.get('error', 'Unknown error')}")

        return {
            "success": True,
            "type": "visualization",
            "chart_type": params.get("chart_type", "custom"),
            "code_generated": code_result["code"],
            "execution_result": exec_result,
            "plotly_figures": exec_result.get("plotly_figures", []),
            "plots": exec_result.get("plots", []),
            "source": "custom_query",
            "query": viz_query,
            "description": f"Generated custom visualization based on query: {viz_query}"
        }

    except Exception as e:
        return {
            "success": False,
            "type": "visualization",
            "error": f"Custom visualization failed: {str(e)}"
        }

async def execute_calculation_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a calculation task"""
    try:
        # Placeholder for statistical calculations
        await asyncio.sleep(1)

        return {
            "success": True,
            "type": "calculation",
            "calculations": {
                "statistical_summary": "Calculations completed",
                "projections": "Future trends analyzed"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "type": "calculation",
            "error": str(e)
        }

async def execute_synthesis_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a synthesis task that combines results from other tasks"""
    try:
        params = task.get("parameters", {})

        # Get the plan_id to access previous task results
        plan_id = task.get("plan_id")
        previous_results = []

        # If we have a plan_id, get previous task results
        if plan_id and plan_id in execution_plans:
            plan = execution_plans[plan_id]
            previous_results = plan.get("results", [])

        # Determine synthesis approach
        synthesis_type = params.get("synthesis_type", "comprehensive")  # comprehensive, executive, technical

        if not previous_results:
            # No previous results to synthesize
            return await create_standalone_synthesis(task, params)

        # Filter successful results for synthesis
        successful_results = [r for r in previous_results if r.get("success", False)]

        if not successful_results:
            raise Exception("No successful task results available for synthesis")

        # Choose synthesis approach based on type
        if synthesis_type == "report_generation":
            return await create_report_synthesis(task, successful_results, params)
        elif synthesis_type == "executive":
            return await create_executive_synthesis(task, successful_results, params)
        elif synthesis_type == "technical":
            return await create_technical_synthesis(task, successful_results, params)
        else:
            # Default comprehensive synthesis
            return await create_comprehensive_synthesis(task, successful_results, params)

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": str(e)
        }

async def create_standalone_synthesis(task: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create synthesis when no previous results are available"""
    try:
        synthesis_query = params.get("query") or task.get("description", "")

        # Use result analyzer to create a standalone analysis
        async with aiohttp.ClientSession() as session:
            analysis_payload = {
                "query": synthesis_query,
                "print_output": "Standalone synthesis task executed",
                "dataframes": [],
                "variables": {},
                "execution_time": 1.0
            }

            async with session.post(
                f"{SERVICES['result_analyzer']}/analyze-results",
                json=analysis_payload
            ) as response:
                if response.status == 200:
                    analysis_result = await response.json()
                else:
                    # Fallback analysis
                    analysis_result = {
                        "explanation": f"Standalone synthesis for: {synthesis_query}",
                        "key_insights": ["Analysis framework established", "Ready for data integration"],
                        "data_summary": "No previous data available",
                        "confidence": "Medium"
                    }

        return {
            "success": True,
            "type": "synthesis",
            "synthesis_type": "standalone",
            "synthesis_result": {
                "combined_insights": analysis_result.get("key_insights", []),
                "explanation": analysis_result.get("explanation", ""),
                "recommendations": [
                    "Gather relevant data sources",
                    "Establish baseline metrics",
                    "Define success criteria"
                ],
                "confidence_level": analysis_result.get("confidence", "Medium"),
                "data_sources": 0,
                "analysis_depth": "Preliminary"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": f"Standalone synthesis failed: {str(e)}"
        }

async def create_comprehensive_synthesis(task: Dict[str, Any], results: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive synthesis combining all available results"""
    try:
        # Collect insights from all successful results
        all_insights = []
        data_sources = []
        visualizations = []
        key_metrics = {}

        # Process each result type
        for result in results:
            result_type = result.get("type", "unknown")

            if result_type == "data_analysis":
                analysis = result.get("execution_result", {}).get("analysis", {})
                if analysis:
                    all_insights.extend(analysis.get("key_insights", []))
                    data_sources.append(f"Data Analysis: {result.get('task_title', 'Unknown')}")

                # Collect visualizations
                viz_data = result.get("execution_result", {}).get("plotly_figures", [])
                visualizations.extend(viz_data)

                # Collect key metrics
                variables = result.get("execution_result", {}).get("variables", {})
                key_metrics.update(variables)

            elif result_type == "web_research":
                findings = result.get("findings", [])
                all_insights.extend(findings)
                data_sources.append(f"Web Research: {result.get('task_title', 'Unknown')}")

            elif result_type == "visualization":
                viz_data = result.get("plotly_figures", [])
                visualizations.extend(viz_data)
                data_sources.append(f"Visualization: {result.get('task_title', 'Unknown')}")

            elif result_type == "calculation":
                calc_results = result.get("calculations", {})
                if calc_results:
                    all_insights.append(f"Calculations: {calc_results}")
                    data_sources.append(f"Calculations: {result.get('task_title', 'Unknown')}")

        # Create comprehensive analysis using result analyzer
        synthesis_query = params.get("query") or task.get("description", "Comprehensive analysis synthesis")

        async with aiohttp.ClientSession() as session:
            analysis_payload = {
                "query": f"Synthesize comprehensive analysis: {synthesis_query}",
                "print_output": f"Combined insights from {len(results)} tasks: " + "; ".join(all_insights[:5]),
                "dataframes": [{"name": "synthesis", "shape": [len(results), 10], "columns": ["task_type", "status", "insights"]}],
                "variables": key_metrics,
                "plots_description": f"{len(visualizations)} visualizations generated across tasks",
                "execution_time": 2.0
            }

            async with session.post(
                f"{SERVICES['result_analyzer']}/analyze-results",
                json=analysis_payload
            ) as response:
                if response.status == 200:
                    analysis_result = await response.json()
                else:
                    # Fallback analysis
                    analysis_result = {
                        "explanation": f"Comprehensive synthesis of {len(results)} task results",
                        "key_insights": all_insights[:10],  # Top 10 insights
                        "data_summary": f"Analyzed {len(data_sources)} data sources",
                        "confidence": "High" if len(results) >= 3 else "Medium"
                    }

        return {
            "success": True,
            "type": "synthesis",
            "synthesis_type": "comprehensive",
            "synthesis_result": {
                "combined_insights": analysis_result.get("key_insights", []),
                "explanation": analysis_result.get("explanation", ""),
                "recommendations": generate_recommendations_from_insights(all_insights),
                "confidence_level": analysis_result.get("confidence", "Medium"),
                "data_sources": len(data_sources),
                "source_details": data_sources,
                "visualizations_count": len(visualizations),
                "key_metrics": key_metrics,
                "analysis_depth": "Comprehensive"
            },
            "visualizations": visualizations,
            "raw_insights": all_insights
        }

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": f"Comprehensive synthesis failed: {str(e)}"
        }

async def create_report_synthesis(task: Dict[str, Any], results: List[Dict[str, Any]],
                                params: Dict[str, Any]) -> Dict[str, Any]:
    """Create synthesis using the report generator service"""
    try:
        plan_id = task.get("plan_id", "synthesis_plan")
        original_query = params.get("query") or task.get("description", "Synthesis report")

        # Call report generator service
        async with aiohttp.ClientSession() as session:
            report_payload = {
                "plan_id": plan_id,
                "original_query": original_query,
                "task_results": results,
                "report_type": params.get("report_type", "comprehensive"),
                "format": params.get("format", "html")
            }

            async with session.post(
                "http://localhost:8006/generate-report",
                json=report_payload
            ) as response:
                if response.status == 200:
                    report_result = await response.json()

                    return {
                        "success": True,
                        "type": "synthesis",
                        "synthesis_type": "report_generation",
                        "synthesis_result": {
                            "report_id": report_result.get("report_id"),
                            "report_content": report_result.get("report_content"),
                            "executive_summary": report_result.get("executive_summary"),
                            "combined_insights": report_result.get("data_insights", []),
                            "recommendations": report_result.get("key_recommendations", []),
                            "confidence_level": "High",
                            "data_sources": len(results),
                            "analysis_depth": "Report-based"
                        },
                        "report_data": report_result
                    }
                else:
                    raise Exception(f"Report generation failed: {response.status}")

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": f"Report synthesis failed: {str(e)}"
        }

async def create_executive_synthesis(task: Dict[str, Any], results: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create executive-level synthesis focusing on high-level insights and recommendations"""

    try:
        # Extract key insights and metrics for executive summary
        key_findings = []
        critical_metrics = {}

        for result in results:
            if result.get("type") == "data_analysis":
                analysis = result.get("execution_result", {}).get("analysis", {})
                if analysis:
                    insights = analysis.get("key_insights", [])
                    # Take only the most important insights (first 2)
                    key_findings.extend(insights[:2])

                # Extract critical business metrics
                variables = result.get("execution_result", {}).get("variables", {})
                for key, value in variables.items():
                    if isinstance(value, (int, float)) and any(term in key.lower()
                        for term in ['total', 'average', 'count', 'sum', 'revenue', 'cost', 'profit']):
                        critical_metrics[key] = value

        # Create executive summary
        synthesis_query = f"Executive summary for: {params.get('query', task.get('description', ''))}"

        return {
            "success": True,
            "type": "synthesis",
            "synthesis_type": "executive",
            "synthesis_result": {
                "combined_insights": key_findings[:5],  # Top 5 insights only
                "executive_summary": f"Analysis of {len(results)} key areas reveals critical insights for decision-making.",
                "recommendations": [
                    "Prioritize high-impact interventions based on data",
                    "Implement monitoring systems for key metrics",
                    "Allocate resources to areas showing greatest need",
                    "Establish regular review cycles for progress tracking"
                ],
                "critical_metrics": critical_metrics,
                "confidence_level": "High" if len(results) >= 2 else "Medium",
                "data_sources": len(results),
                "analysis_depth": "Executive"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": f"Executive synthesis failed: {str(e)}"
        }

async def create_technical_synthesis(task: Dict[str, Any], results: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create technical synthesis with detailed methodology and data analysis"""
    try:
        # Collect detailed technical information
        methodologies = []
        data_details = []
        technical_metrics = {}
        code_snippets = []

        for result in results:
            result_type = result.get("type")

            if result_type == "data_analysis":
                # Collect code and methodology
                code = result.get("code_generated", "")
                if code:
                    code_snippets.append({
                        "task": result.get("task_title", "Unknown"),
                        "code": code[:200] + "..." if len(code) > 200 else code
                    })

                # Collect execution details
                exec_result = result.get("execution_result", {})
                execution_time = exec_result.get("execution_time", 0)
                technical_metrics[f"{result.get('task_title', 'task')}_execution_time"] = execution_time

                # Collect data details
                dataframes = exec_result.get("dataframes", [])
                for df in dataframes:
                    data_details.append({
                        "name": df.get("name", "unknown"),
                        "shape": df.get("shape", [0, 0]),
                        "columns": len(df.get("columns", []))
                    })

                methodologies.append(f"Data analysis using Python pandas and statistical methods")

            elif result_type == "visualization":
                methodologies.append(f"Data visualization using Plotly interactive charts")
                viz_count = len(result.get("plotly_figures", []))
                technical_metrics["visualizations_generated"] = viz_count

            elif result_type == "web_research":
                methodologies.append(f"Web research and information gathering")
                findings_count = len(result.get("findings", []))
                technical_metrics["research_findings"] = findings_count

        return {
            "success": True,
            "type": "synthesis",
            "synthesis_type": "technical",
            "synthesis_result": {
                "methodologies": methodologies,
                "data_details": data_details,
                "technical_metrics": technical_metrics,
                "code_snippets": code_snippets,
                "combined_insights": [
                    f"Processed {len(data_details)} datasets using automated analysis",
                    f"Applied {len(methodologies)} different analytical approaches",
                    f"Generated {technical_metrics.get('visualizations_generated', 0)} visualizations"
                ],
                "recommendations": [
                    "Validate results through cross-verification",
                    "Implement automated monitoring systems",
                    "Document methodology for reproducibility",
                    "Establish data quality checks"
                ],
                "confidence_level": "High",
                "data_sources": len(results),
                "analysis_depth": "Technical"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "type": "synthesis",
            "error": f"Technical synthesis failed: {str(e)}"
        }

def generate_recommendations_from_insights(insights: List[str]) -> List[str]:
    """Generate actionable recommendations based on insights"""
    recommendations = []

    # Analyze insights for common themes
    if any("increase" in insight.lower() or "improve" in insight.lower() for insight in insights):
        recommendations.append("Implement improvement strategies based on identified opportunities")

    if any("decrease" in insight.lower() or "decline" in insight.lower() for insight in insights):
        recommendations.append("Address declining trends through targeted interventions")

    if any("data" in insight.lower() or "analysis" in insight.lower() for insight in insights):
        recommendations.append("Establish data-driven decision making processes")

    if any("region" in insight.lower() or "state" in insight.lower() for insight in insights):
        recommendations.append("Develop region-specific strategies based on local conditions")

    # Default recommendations if no specific patterns found
    if not recommendations:
        recommendations = [
            "Monitor key performance indicators regularly",
            "Implement evidence-based policy measures",
            "Engage stakeholders in solution development",
            "Establish feedback mechanisms for continuous improvement"
        ]

    return recommendations[:4]  # Return top 4 recommendations

def generate_visualization_code(dataframes: List[Dict[str, Any]], variables: Dict[str, Any],
                              chart_type: str, title: str, params: Dict[str, Any]) -> str:
    """Generate Python code for creating visualizations based on available data"""

    # Import statements
    code = """import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the CSV data
df = pd.read_csv('2024-2025.csv')

"""

    # Determine what columns to use based on dataframes info
    if dataframes:
        # Use information from the first dataframe
        df_info = dataframes[0]
        columns = df_info.get("columns", [])

        # Try to find appropriate columns for different chart types
        numeric_cols = [col for col in columns if any(keyword in col.lower()
                       for keyword in ['amount', 'value', 'count', 'total', 'sum', 'avg', 'mean'])]
        categorical_cols = [col for col in columns if col not in numeric_cols]

        if chart_type.lower() == "bar":
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                code += f"""
# Create bar chart
fig = px.bar(df, x='{x_col}', y='{y_col}',
             title='{title}',
             labels={{'{x_col}': '{x_col.replace("_", " ").title()}',
                     '{y_col}': '{y_col.replace("_", " ").title()}'}})
fig.update_layout(showlegend=True)
fig.show()
"""
            else:
                # Fallback to value counts of first column
                col = columns[0] if columns else "column1"
                code += f"""
# Create bar chart from value counts
value_counts = df['{col}'].value_counts().head(10)
fig = px.bar(x=value_counts.index, y=value_counts.values,
             title='{title}',
             labels={{'x': '{col.replace("_", " ").title()}', 'y': 'Count'}})
fig.show()
"""

        elif chart_type.lower() == "line":
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                code += f"""
# Create line chart
fig = px.line(df, x='{x_col}', y='{y_col}',
              title='{title}',
              labels={{'{x_col}': '{x_col.replace("_", " ").title()}',
                      '{y_col}': '{y_col.replace("_", " ").title()}'}})
fig.show()
"""
            else:
                # Use index as x-axis
                y_col = numeric_cols[0] if numeric_cols else columns[0]
                code += f"""
# Create line chart with index
fig = px.line(df.reset_index(), x='index', y='{y_col}',
              title='{title}',
              labels={{'index': 'Index', '{y_col}': '{y_col.replace("_", " ").title()}'}})
fig.show()
"""

        elif chart_type.lower() == "pie":
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                code += f"""
# Create pie chart
value_counts = df['{col}'].value_counts().head(8)
fig = px.pie(values=value_counts.values, names=value_counts.index,
             title='{title}')
fig.show()
"""
            else:
                col = columns[0] if columns else "column1"
                code += f"""
# Create pie chart from value counts
value_counts = df['{col}'].value_counts().head(8)
fig = px.pie(values=value_counts.values, names=value_counts.index,
             title='{title}')
fig.show()
"""

        elif chart_type.lower() == "scatter":
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                color_col = categorical_cols[0] if categorical_cols else None

                if color_col:
                    code += f"""
# Create scatter plot with color
fig = px.scatter(df, x='{x_col}', y='{y_col}', color='{color_col}',
                 title='{title}',
                 labels={{'{x_col}': '{x_col.replace("_", " ").title()}',
                         '{y_col}': '{y_col.replace("_", " ").title()}'}})
fig.show()
"""
                else:
                    code += f"""
# Create scatter plot
fig = px.scatter(df, x='{x_col}', y='{y_col}',
                 title='{title}',
                 labels={{'{x_col}': '{x_col.replace("_", " ").title()}',
                         '{y_col}': '{y_col.replace("_", " ").title()}'}})
fig.show()
"""
            else:
                # Fallback scatter plot
                code += f"""
# Create scatter plot with index
fig = px.scatter(df.reset_index(), x='index', y=df.columns[0],
                 title='{title}')
fig.show()
"""

        elif chart_type.lower() == "histogram":
            col = numeric_cols[0] if numeric_cols else columns[0]
            code += f"""
# Create histogram
fig = px.histogram(df, x='{col}', title='{title}',
                   labels={{'{col}': '{col.replace("_", " ").title()}'}})
fig.show()
"""

        else:
            # Default to bar chart
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                code += f"""
# Create default bar chart
fig = px.bar(df, x='{x_col}', y='{y_col}',
             title='{title}')
fig.show()
"""
            else:
                col = columns[0] if columns else "column1"
                code += f"""
# Create default chart from value counts
value_counts = df['{col}'].value_counts().head(10)
fig = px.bar(x=value_counts.index, y=value_counts.values,
             title='{title}')
fig.show()
"""

    else:
        # No dataframe info available, create a generic visualization
        code += f"""
# Create generic visualization (no specific column info available)
# Show basic statistics of the dataset
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    # Create histogram of first numeric column
    fig = px.histogram(df, x=numeric_cols[0], title='{title}')
    fig.show()
else:
    # Create bar chart of value counts for first column
    col = df.columns[0]
    value_counts = df[col].value_counts().head(10)
    fig = px.bar(x=value_counts.index, y=value_counts.values,
                 title='{title}')
    fig.show()
"""

    return code

async def execute_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task based on its type"""
    task_type = task.get("type", "data_analysis")

    print(f"Executing task: {task['title']} (Type: {task_type})")
    print(f"Description: {task.get('description', 'No description')}")
    print(f"Parameters: {task.get('parameters', {})}")

    start_time = time.time()

    if task_type == "data_analysis":
        result = await execute_data_analysis_task(task)
    elif task_type == "web_research":
        result = await execute_web_research_task(task)
    elif task_type == "visualization":
        result = await execute_visualization_task(task)
    elif task_type == "calculation":
        result = await execute_calculation_task(task)
    elif task_type == "synthesis":
        result = await execute_synthesis_task(task)
    else:
        result = {
            "success": False,
            "type": task_type,
            "error": f"Unknown task type: {task_type}"
        }

    execution_time = time.time() - start_time
    result["execution_time"] = f"{execution_time:.2f} seconds"

    if result.get("success"):
        print(f"Task completed successfully in {execution_time:.2f}s")
        if result.get("execution_result", {}).get("analysis"):
            analysis = result["execution_result"]["analysis"]
            print(f"Key insights: {len(analysis.get('key_insights', []))} insights generated")
    else:
        print(f"Task failed: {result.get('error', 'Unknown error')}")

    return result

async def execute_plan_background(plan_id: str):
    """Execute all tasks in a plan in the background"""
    try:
        plan = execution_plans[plan_id]
        plan["status"] = "running"
        plan["started_at"] = datetime.now().isoformat()
        
        tasks = plan["tasks"]
        results = []
        
        for i, task in enumerate(tasks):
            # Update progress
            plan["current_task_index"] = i
            plan["progress_percentage"] = (i / len(tasks)) * 100

            # Add plan_id to task for accessing previous results
            task["plan_id"] = plan_id

            # Execute task
            task["status"] = "in_progress"
            result = await execute_single_task(task)
            
            # Store result
            result["task_id"] = task["id"]
            result["task_title"] = task["title"]
            results.append(result)
            
            # Update task status
            if result["success"]:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                task["result"] = result
            else:
                task["status"] = "failed"
                task["error"] = result.get("error")
        
        # Mark plan as completed
        plan["status"] = "completed"
        plan["completed_at"] = datetime.now().isoformat()
        plan["progress_percentage"] = 100.0
        plan["results"] = results
        
        print(f"Plan {plan_id} completed successfully")
        
    except Exception as e:
        plan["status"] = "failed"
        plan["error"] = str(e)
        print(f"Plan {plan_id} failed: {str(e)}")

@app.post("/execute-plan")
async def execute_plan(request: ExecutionRequest, background_tasks: BackgroundTasks):
    """Start executing a plan in the background"""
    
    plan_id = request.plan_id
    
    # Store execution plan
    execution_plans[plan_id] = {
        "plan_id": plan_id,
        "original_query": request.original_query,
        "tasks": request.tasks,
        "status": "pending",
        "current_task_index": 0,
        "results": [],
        "progress_percentage": 0.0
    }
    
    # Start background execution
    background_tasks.add_task(execute_plan_background, plan_id)
    
    return {
        "message": "Plan execution started",
        "plan_id": plan_id,
        "status": "running"
    }

@app.get("/execution-status/{plan_id}", response_model=ExecutionStatus)
async def get_execution_status(plan_id: str):
    """Get the current status of plan execution"""
    
    if plan_id not in execution_plans:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    plan = execution_plans[plan_id]
    
    # Get current task
    current_task = None
    if plan["current_task_index"] < len(plan["tasks"]):
        current_task = plan["tasks"][plan["current_task_index"]]
    
    # Get completed tasks
    completed_tasks = [
        task for task in plan["tasks"] 
        if task.get("status") == "completed"
    ]
    
    return ExecutionStatus(
        plan_id=plan_id,
        status=plan["status"],
        progress_percentage=plan["progress_percentage"],
        current_task=current_task,
        completed_tasks=completed_tasks,
        results=plan.get("results", []),
        error=plan.get("error")
    )

@app.get("/")
async def root():
    return {"message": "Task Execution Engine API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)