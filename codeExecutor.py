from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io
import sys
import base64
import time
import traceback
import json
import threading
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional
import uvicorn
import requests
import numpy as np

app = FastAPI(title="Code Executor", description="Execute Python code safely for CSV analysis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models are a core feature of Pydantic, a Python library used to validate, parse, and manage data. 
# They are especially popular with FastAPI for request and response handling.

class ExecutionRequest(BaseModel):  
    code: str
    csv_file: str = "2024-2025.csv"
    include_analysis: bool = True
    original_query: Optional[str] = None

class AnalysisResult(BaseModel):
    explanation: str
    key_insights: List[str]
    data_summary: str
    confidence: str

class ExecutionResponse(BaseModel):
    success: bool
    execution_time: float
    print_output: str
    dataframes: List[Dict[str, Any]]
    plots: List[Dict[str, str]]  # Matplotlib plots as base64 images
    plotly_figures: List[Dict[str, Any]]  # Plotly figures as JSON
    variables: Dict[str, Any]
    analysis: Optional[AnalysisResult] = None  # Natural language explanation
    error: Optional[str]

class TimeoutException(Exception):
    pass

def execute_with_timeout(code, globals_dict, locals_dict, timeout_seconds=30):
    """Execute code with timeout using threading"""
    result = {"success": False, "error": None}

    def target():
        try:
            exec(code, globals_dict, locals_dict)
            result["success"] = True
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutException("Code execution timed out")

    if result["error"]:
        raise result["error"]

    return result["success"]

def capture_dataframes(local_vars: Dict) -> List[Dict[str, Any]]:
    """Extract DataFrames from execution context"""
    dataframes = []
    
    for name, value in local_vars.items():
        if isinstance(value, pd.DataFrame) and not name.startswith('_'):
            try:
                # Convert DataFrame to JSON-serializable format
                df_info = {
                    "name": name,
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                    "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()},
                    "head": value.head(10).to_dict('records'),
                    "summary": {
                        "total_rows": len(value),
                        "total_columns": len(value.columns),
                        "memory_usage": f"{value.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                    }
                }
                
                # Add basic statistics for numeric columns
                numeric_cols = value.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_info["statistics"] = value[numeric_cols].describe().to_dict()
                
                dataframes.append(df_info)
            except Exception as e:
                print(f"Error processing DataFrame {name}: {e}")
    
    return dataframes

def capture_plots() -> List[Dict[str, str]]:
    """Capture matplotlib plots as base64 images"""
    plots = []
    
    # Get all figure numbers
    fig_nums = plt.get_fignums()
    
    for i, fig_num in enumerate(fig_nums):
        try:
            fig = plt.figure(fig_num)
            
            # Save plot to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plots.append({
                "name": f"plot_{i+1}",
                "image": f"data:image/png;base64,{image_base64}"
            })
            
            buffer.close()
        except Exception as e:
            print(f"Error capturing plot {fig_num}: {e}")
    
    # Clear all figures to prevent memory leaks
    plt.close('all')
    
    return plots

def capture_plotly_figures(execution_locals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Capture Plotly figures as JSON for interactive rendering"""
    plotly_figures = []

    # Look for Plotly figure objects in the execution namespace
    for var_name, var_value in execution_locals.items():
        if not var_name.startswith('_'):
            try:
                # Check if it's a Plotly figure
                if hasattr(var_value, 'to_json') and hasattr(var_value, 'data'):
                    # It's a Plotly figure
                    fig_json = var_value.to_json()
                    fig_dict = var_value.to_dict()

                    plotly_figures.append({
                        "name": var_name,
                        "type": "plotly",
                        "figure": fig_dict,
                        "json": fig_json
                    })
                    print(f"Captured Plotly figure: {var_name}")

            except Exception as e:
                # Silently skip non-Plotly objects
                pass

    return plotly_figures

def clean_for_json(obj):
    """Clean data structures for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

async def get_result_analysis(query: str, execution_response: Dict[str, Any]) -> Optional[AnalysisResult]:
    """Get natural language analysis of execution results"""
    try:
        analysis_request = {
            "query": query,
            "print_output": execution_response.get("print_output", ""),
            "dataframes": execution_response.get("dataframes", []),
            "variables": execution_response.get("variables", {}),
            "execution_time": execution_response.get("execution_time", 0.0)
        }

        # Add plot descriptions if available
        plots_desc = []
        if execution_response.get("plots"):
            plots_desc.append(f"{len(execution_response['plots'])} matplotlib plots generated")
        if execution_response.get("plotly_figures"):
            plots_desc.append(f"{len(execution_response['plotly_figures'])} interactive Plotly charts created")

        if plots_desc:
            analysis_request["plots_description"] = "; ".join(plots_desc)

        # Call analysis service
        response = requests.post(
            "http://localhost:8003/analyze-results",
            json=analysis_request,
            timeout=30
        )

        if response.status_code == 200:
            analysis_data = response.json()
            return AnalysisResult(**analysis_data)
        else:
            print(f"Analysis service error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Failed to get result analysis: {e}")
        return None

def extract_variables(local_vars: Dict) -> Dict[str, Any]:
    """Extract simple variables (numbers, strings, lists) from execution context"""
    variables = {}
    
    for name, value in local_vars.items():
        if not name.startswith('_') and not callable(value):
            try:
                # Only include JSON-serializable types
                if isinstance(value, (int, float, str, bool, list, dict)):
                    variables[name] = value
                elif isinstance(value, np.integer):
                    variables[name] = int(value)
                elif isinstance(value, np.floating):
                    variables[name] = float(value)
                elif isinstance(value, pd.Series):
                    # Convert small Series to dict
                    if len(value) <= 100:
                        variables[name] = value.to_dict()
            except Exception:
                # Skip variables that can't be serialized
                pass
    
    return variables

@app.post("/execute-code", response_model=ExecutionResponse)
async def execute_code(request: ExecutionRequest):
    """Execute Python code and return results"""
    return await _execute_code_internal(request)

@app.post("/execute-with-analysis", response_model=ExecutionResponse)
async def execute_with_analysis(request: ExecutionRequest):
    """Execute Python code and return results with natural language analysis"""
    request.include_analysis = True
    return await _execute_code_internal(request)

async def _execute_code_internal(request: ExecutionRequest):
    """Execute Python code and return results"""
    
    start_time = time.time()
    
    # Prepare execution environment
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Set up execution context with common imports
    execution_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'np': np,
        'plt': plt,
        'print': print,
        'len': len,
        'range': range,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
    }
    
    execution_locals = {}
    
    try:
        # Execute code with output capture and timeout
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            execute_with_timeout(request.code, execution_globals, execution_locals, 30)
        
        # Capture results
        execution_time = time.time() - start_time
        print_output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        # Extract different types of outputs
        dataframes = capture_dataframes(execution_locals)
        plots = capture_plots()
        plotly_figures = capture_plotly_figures(execution_locals)
        variables = extract_variables(execution_locals)

        # Create base response and clean for JSON serialization
        response_data = clean_for_json({
            "success": True,
            "execution_time": round(execution_time, 3),
            "print_output": print_output,
            "dataframes": dataframes,
            "plots": plots,
            "plotly_figures": plotly_figures,
            "variables": variables,
            "error": error_output if error_output else None
        })

        # Add analysis if requested
        analysis = None
        if request.include_analysis and request.original_query:
            analysis = await get_result_analysis(request.original_query, response_data)

        return ExecutionResponse(
            **response_data,
            analysis=analysis
        )

    except TimeoutException:
        return ExecutionResponse(
            success=False,
            execution_time=30.0,
            print_output="",
            dataframes=[],
            plots=[],
            plotly_figures=[],
            variables={},
            analysis=None,
            error="Code execution timed out (30 seconds limit)"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_output = stderr_capture.getvalue()
        
        # Get detailed error information
        error_details = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        if error_output:
            error_details += f"\n\nStderr:\n{error_output}"
        
        return ExecutionResponse(
            success=False,
            execution_time=round(execution_time, 3),
            print_output=stdout_capture.getvalue(),
            dataframes=[],
            plots=[],
            plotly_figures=[],
            variables={},
            analysis=None,
            error=error_details
        )

@app.get("/")
async def root():
    return {"message": "Code Executor API is running"}

@app.get("/test-execution")
async def test_execution():
    """Test the execution environment"""
    test_code = """
import pandas as pd
import numpy as np

# Create test data
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
test_df = pd.DataFrame(data)
result = test_df.sum()
print("Test execution successful!")
print(f"DataFrame shape: {test_df.shape}")
"""
    
    request = ExecutionRequest(code=test_code, csv_file="test.csv")
    return await execute_code(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
