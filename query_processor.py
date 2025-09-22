"""
AI Query Processing Module for INGRES ChatBot
Handles natural language to SQL conversion and visualization selection using OpenAI ChatGPT
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sqlalchemy import text
from database_manager import DatabaseManager
from visualisation_tools import PLOT_FUNCTIONS, create_histogram, create_line_chart, create_pie_chart, create_bar_chart, create_scatter_plot, create_box_plot, create_violin_plot, create_heatmap, create_density_contour, create_density_heatmap, create_area_chart, create_funnel_chart, create_timeline_chart, create_sunburst_chart, create_treemap_chart, create_icicle_chart, create_parallel_coordinates, create_parallel_categories, create_choropleth, create_scatter_geo
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes natural language queries and converts them to SQL and visualizations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.llm = None
        self._initialize_llm()
        self.database_schema = self._get_database_schema()
    
    def _initialize_llm(self):
        """Initialize OpenAI ChatGPT LLM"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.llm = ChatOpenAI(
                model="gpt-4o-mini-2024-07-18",
                openai_api_key=api_key,
                temperature=0.1
            )
            logger.info("OpenAI ChatGPT LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _get_database_schema(self) -> Dict:
        """Get comprehensive database schema information"""
        schema_info = {}
        
        try:
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'groundwater_data_%'
                """))
                
                table_names = [row[0] for row in result]
                
                for table_name in table_names:
                    schema = self.db_manager.get_table_schema(table_name)
                    schema_info[table_name] = schema
                
            logger.info(f"Retrieved schema for {len(schema_info)} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return {}

    def _get_common_columns(self) -> List[str]:
        """Get columns that exist in all groundwater tables with compatible types"""
        try:
            if not self.database_schema:
                return []
                
            all_columns = {}
            
            # Get columns from each table
            for table_name, schema in self.database_schema.items():
                for col in schema['columns']:
                    col_name = col['column_name']
                    col_type = col['data_type']
                    
                    if col_name not in all_columns:
                        all_columns[col_name] = {'types': set(), 'count': 0}
                    
                    all_columns[col_name]['types'].add(col_type)
                    all_columns[col_name]['count'] += 1
            
            # Find columns that exist in all tables with same type
            total_tables = len(self.database_schema)
            common_columns = []
            
            for col_name, col_info in all_columns.items():
                if col_info['count'] == total_tables and len(col_info['types']) == 1:
                    common_columns.append(col_name)
            
            logger.info(f"Found {len(common_columns)} common columns across all tables")
            return sorted(common_columns)
            
        except Exception as e:
            logger.error(f"Error finding common columns: {e}")
            return []

    def _get_union_compatible_columns(self) -> Dict[str, List[str]]:
        """Get groups of columns that can be safely used in UNION operations"""
        try:
            compatible_groups = {
                'basic_info': ['s_no', 'state', 'district', 'assessment_unit'],
                'rainfall': ['rainfall_mm_c', 'rainfall_mm_nc', 'rainfall_mm_pq', 'rainfall_mm_total'],
                'extractable_resources': ['annual_extractable_ground_water_resource_ham_total'],
                'recharge_data': ['ground_water_recharge_ham_total']
            }
            
            # Verify these columns exist in all tables with compatible types
            verified_groups = {}
            
            for group_name, columns in compatible_groups.items():
                verified_columns = []
                for col in columns:
                    # Check if column exists in all tables
                    exists_in_all = True
                    types = set()
                    
                    for table_name, schema in self.database_schema.items():
                        found = False
                        for table_col in schema['columns']:
                            if table_col['column_name'] == col:
                                found = True
                                types.add(table_col['data_type'])
                                break
                        
                        if not found:
                            exists_in_all = False
                            break
                    
                    # Include column if it exists in all tables with compatible types
                    if exists_in_all and len(types) <= 2:  # Allow some type flexibility (e.g., int/float)
                        verified_columns.append(col)
                
                if verified_columns:
                    verified_groups[group_name] = verified_columns
            
            return verified_groups
            
        except Exception as e:
            logger.error(f"Error getting union compatible columns: {e}")
            return {}

    def _get_type_mismatch_columns(self) -> Dict[str, Dict[str, str]]:
        """Identify columns with type mismatches across tables"""
        type_mismatches = {}
        
        try:
            # Get all column names
            all_columns = set()
            for schema in self.database_schema.values():
                for col in schema['columns']:
                    all_columns.add(col['column_name'])
            
            # Check each column across all tables
            for col_name in all_columns:
                table_types = {}
                for table_name, schema in self.database_schema.items():
                    for table_col in schema['columns']:
                        if table_col['column_name'] == col_name:
                            table_types[table_name] = table_col['data_type']
                            break
                
                # If this column appears in multiple tables with different types
                if len(table_types) > 1:
                    unique_types = set(table_types.values())
                    if len(unique_types) > 1:
                        type_mismatches[col_name] = table_types
            
            return type_mismatches
            
        except Exception as e:
            logger.error(f"Error detecting type mismatches: {e}")
            return {}
    
    def _create_schema_context(self) -> str:
        """Create a formatted schema context for the LLM"""
        schema_context = "Database Schema Information:\n\n"
        
        for table_name, schema in self.database_schema.items():
            year = table_name.replace('groundwater_data_', '').replace('_', '-')
            schema_context += f"Table: {table_name} (Year: {year})\n"
            schema_context += "Columns:\n"
            for col in schema['columns']:
                col_name = col['column_name']
                col_type = col['data_type']
                schema_context += f"  - {col_name} ({col_type})\n"
            schema_context += "\n"
        
        # Add common columns information
        common_columns = self._get_common_columns()
        if common_columns:
            schema_context += "COMMON COLUMNS (Safe for UNION operations):\n"
            for col in common_columns[:20]:  # Limit to first 20 for brevity
                schema_context += f"  - {col}\n"
            if len(common_columns) > 20:
                schema_context += f"  ... and {len(common_columns) - 20} more\n"
            schema_context += "\n"
        
        # Add UNION-compatible column groups
        union_groups = self._get_union_compatible_columns()
        if union_groups:
            schema_context += "RECOMMENDED COLUMN GROUPS FOR UNION OPERATIONS:\n"
            for group_name, columns in union_groups.items():
                schema_context += f"  {group_name}: {', '.join(columns)}\n"
            schema_context += "\n"
        
        # Add type mismatch warnings
        type_mismatches = self._get_type_mismatch_columns()
        if type_mismatches:
            schema_context += "⚠️  CRITICAL TYPE MISMATCH WARNINGS FOR UNION OPERATIONS:\n"
            for col_name, table_types in type_mismatches.items():
                schema_context += f"  {col_name}:\n"
                for table, dtype in table_types.items():
                    schema_context += f"    - {table}: {dtype}\n"
            schema_context += "\n🚨 ALWAYS cast these columns to compatible types in UNION operations!\n"
            schema_context += "   Example: CAST(district AS TEXT), CAST(rainfall_mm_total AS NUMERIC)\n\n"
        
        schema_context += """
Domain Knowledge Context:
- This database contains groundwater resource assessment data from India
- Data is organized by year (2012-2013, 2016-2017, etc.)
- Each row represents an assessment unit (Block/Mandal/Taluk)
- Key metrics include:
  * Rainfall data (mm)
  * Geographical area (ha)
  * Ground water recharge (ham)
  * Ground water extraction (ha.m)
  * Stage of ground water extraction (%)
- Suffixes: _C (Consolidated), _NC (Non-Consolidated), _PQ (Pre-Quaternary), _Total
- Assessment units are categorized as Safe, Semi-Critical, Critical, or Over-Exploited
- ham = hectare-meters, ha = hectares
"""
        return schema_context
    
    def generate_sql_query(self, user_question: str) -> str:
        """Generate SQL query from natural language question with feedback loop"""
        schema_context = self._create_schema_context()
        max_loops = 5
        loop_count = 0
        exploration_data = []
        
        while loop_count < max_loops:
            loop_count += 1
            
            # Create exploration context from previous queries
            exploration_context = ""
            if exploration_data:
                exploration_context = "\nPrevious exploration data:\n"
                for i, data in enumerate(exploration_data, 1):
                    status_indicator = "✓" if data['status'] == 'success' else "✗"
                    exploration_context += f"Query {i} ({status_indicator}): {data['query']}\n"
                    exploration_context += f"Results: {data['results']}\n"
                    exploration_context += f"Explanation: {data['explanation']}\n\n"
            
            # CRITICAL: Use the improved prompt template with UNION fixes
            sql_prompt = PromptTemplate(
                input_variables=["schema", "question", "exploration_context", "loop_count", "max_loops"],
                template="""
    You are an expert SQL query generator and agricultural data analyst for a groundwater resource database.
    {schema}

    {exploration_context}

    User Question: {question}

    Loop Count: {loop_count}/{max_loops}

    CRITICAL INSTRUCTIONS:
    1. If this is your FINAL loop ({loop_count} == {max_loops}), you MUST generate the FINAL SQL query that answers the user's question and MUST set need_exploration to false. Do not explore in the final loop.
    2. If you need to explore data first OR if previous queries had errors and this is NOT the final loop, generate an improved query and set need_exploration to true.
    3. LEARN FROM PREVIOUS QUERY ERRORS: If previous queries failed, understand why and avoid similar mistakes - pay special attention to table alias errors
    4. For UNION operations, NEVER use SELECT * - ONLY use columns from the "COMMON COLUMNS" or "RECOMMENDED COLUMN GROUPS" section above
    5. When combining data from multiple years, use only the columns listed in the schema as common or compatible
    6. Use appropriate table names and column names from the schema
    7. Include proper WHERE clauses, JOINs, and aggregations as needed
    8. For time-based queries, consider which year's data to use
    9. Ensure the query is PostgreSQL compatible
    10. Use proper column aliases for better readability
    11. Limit exploration queries to 20 results, final queries to reasonable numbers
    12. All state names with two words have spaces like "MADHYA PRADESH" except "TAMILNADU"
    13. You MUST always return a valid SQL query that can be executed
    14. Never return empty strings or invalid queries
    15. In the final loop, use all available information from previous explorations to create the best possible query
    16. Double-check table aliases in JOINs - ensure columns are referenced from the correct table alias
    17. While doing union keep in mind that the table groundwater_data_2012_2013 has district as double precision(they are null) whereas in other tables it is text. So you will have to cast it appropriately.

    **RESPONSE FORMAT REQUIREMENT:**
    You MUST respond with a valid JSON object in the following format:
    {{
        "sql_query": "YOUR SQL QUERY HERE",
        "need_exploration": true/false,
        "explanation": "Brief explanation of what this query does"
    }}

    Do NOT include any text outside the JSON object. Do NOT wrap the JSON in code blocks or markdown.
        """
            )
            
            try:
                chain = sql_prompt | self.llm
                response = chain.invoke({
                    "schema": schema_context,
                    "question": user_question,
                    "exploration_context": exploration_context,
                    "loop_count": loop_count,
                    "max_loops": max_loops
                })
                
                # Parse the JSON response
                response_content = response.content.strip()
                
                # Clean up markdown code blocks if present
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                elif response_content.startswith('```'):
                    response_content = response_content.replace('```', '').strip()
                
                try:
                    import json
                    parsed_response = json.loads(response_content)
                    sql_query = parsed_response.get('sql_query', '').strip()
                    need_exploration = parsed_response.get('need_exploration', False)
                    explanation = parsed_response.get('explanation', '')
                    
                    logger.info(f"Loop {loop_count}: {explanation}")
                    logger.info(f"Generated SQL query: {sql_query}")
                    logger.info(f"Need exploration: {need_exploration}")
                    
                    if not sql_query:
                        logger.error("Empty SQL query received from LLM")
                        continue
                    
                    # CRITICAL FIX: Detect and prevent SELECT * UNION operations
                    if 'select *' in sql_query.lower() and 'union' in sql_query.lower():
                        logger.error("Detected SELECT * with UNION - this will cause type mismatch!")
                        if loop_count < max_loops:
                            # Add error to exploration data and continue
                            exploration_data.append({
                                'query': sql_query,
                                'results': "Query contains SELECT * with UNION - this causes type mismatch errors. Use explicit column selection with casting: SELECT CAST(state AS TEXT), CAST(district AS TEXT), ...",
                                'explanation': explanation,
                                'status': 'failed'
                            })
                            continue
                    
                    # If this is the final query or no exploration needed, test it first
                    if not need_exploration or loop_count >= max_loops:
                        # Test the query to see if it works before returning it
                        success, test_df, test_message = self.execute_query_safely(sql_query)
                        if success:
                            logger.info(f"Query validated successfully, returning after {loop_count} loops")
                            return sql_query
                        elif loop_count >= max_loops:
                            # If this is the final loop and query failed, return it anyway (with error logged)
                            logger.error(f"Final query failed validation but returning after {loop_count} loops: {test_message}")
                            return sql_query
                        else:
                            logger.warning(f"Query validation failed: {test_message}")
                            # Add the failed query to exploration data and continue to next iteration
                            exploration_data.append({
                                'query': sql_query,
                                'results': f"Query validation failed: {test_message}",
                                'explanation': explanation,
                                'status': 'failed'
                            })
                            continue
                    
                    # Execute exploration query and store results
                    try:
                        success, exploration_df, message = self.execute_query_safely(sql_query)
                        if success and not exploration_df.empty:
                            # Convert results to string format for next iteration
                            if len(exploration_df) <= 10:
                                results_str = exploration_df.to_string(index=False, max_rows=10)
                            else:
                                results_str = exploration_df.head(10).to_string(index=False)
                                results_str += f"\n... ({len(exploration_df)} total rows)"
                            
                            exploration_data.append({
                                'query': sql_query,
                                'results': results_str,
                                'explanation': explanation,
                                'status': 'success'
                            })
                            
                            logger.info(f"Exploration query executed successfully, got {len(exploration_df)} rows")
                        else:
                            logger.warning(f"Exploration query failed or returned no data: {message}")
                            # Continue to next iteration with this information
                            exploration_data.append({
                                'query': sql_query,
                                'results': f"Query failed: {message}",
                                'explanation': explanation,
                                'status': 'failed'
                            })
                            
                    except Exception as e:
                        logger.error(f"Error executing exploration query: {e}")
                        exploration_data.append({
                            'query': sql_query,
                            'results': f"Execution error: {str(e)}",
                            'explanation': explanation,
                            'status': 'error'
                        })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response_content}")
                    
                    # Continue to next iteration to let LLM try again
                    continue
                        
            except Exception as e:
                logger.error(f"Error in generate_sql_query loop {loop_count}: {e}")
                continue
        
        # This should never be reached as the LLM is instructed to always provide a final query in the last loop
        logger.error("Feedback loop completed without final query - this should not happen")
        raise Exception("LLM failed to provide final query within the specified loops")

    def _execute_exploration_query(self, sql_query: str) -> Tuple[bool, str]:
        """Execute an exploration query and return formatted results"""
        try:
            success, df, message = self.execute_query_safely(sql_query)
            if success and not df.empty:
                if len(df) <= 20:
                    return True, df.to_string(index=False)
                else:
                    result_str = df.head(20).to_string(index=False)
                    result_str += f"\n... (showing first 20 of {len(df)} rows)"
                    return True, result_str
            else:
                return False, f"Query failed or returned no data: {message}"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def generate_visualization(self, user_question: str, data: pd.DataFrame) -> Optional[go.Figure]:
        """Generate a visualization using LLM to select chart type"""
        
        logger.info(f"Checking for visualization keywords in: '{user_question}'")
        logger.info(f"Data empty: {data.empty}")
        logger.info(f"Data shape: {data.shape if not data.empty else 'N/A'}")
        
        if data.empty:
            logger.info("No visualization will be generated - data is empty")
            return None
        
        # Always try to create visualization for numeric data
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        logger.info(f"Numeric columns: {list(numeric_cols)}")
        logger.info(f"Categorical columns: {list(categorical_cols)}")
        
        # If we have suitable data for visualization, proceed even without explicit keywords
        if len(numeric_cols) == 0:
            logger.info("No numeric columns found for visualization")
            return None
        
        data_sample = data.head(10).to_dict(orient='records')
        columns_info = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        viz_prompt = PromptTemplate(
            input_variables=["question", "data_sample", "columns", "plot_functions"],
            template="""
You are an expert data visualization assistant. Based on the user query and data, select the best Plotly chart function from the provided tools to visualize the data. Return a JSON object with the function name and arguments.

User Query: {question}
Data Sample (first 10 rows): {data_sample}
Columns and Types: {columns}
Available Plot Functions: {plot_functions}

Instructions:
1. Choose the most appropriate visualization function from the provided tools.
2. Ensure the selected columns exist in the data and match the required data types (e.g., numeric for y_column, categorical for names_column).
3. For geographic charts (choropleth, scatter_geo), only select if 'state' or 'district' columns are present (for choropleth) or 'latitude' and 'longitude' (for scatter_geo).
4. Return JSON: {{"function_name": "name", "arguments": {{...}}}}
5. If no visualization is suitable, return {{"function_name": "none"}}
6. Keep arguments minimal and relevant to the query.
7. Ensure the 'data' argument is the same as the provided data sample.
"""
        )
        
        try:
            chain = viz_prompt | self.llm
            response = chain.invoke({
                "question": user_question,
                "data_sample": json.dumps(data_sample),
                "columns": json.dumps(columns_info),
                "plot_functions": json.dumps(PLOT_FUNCTIONS, indent=2)
            })
            logger.info(f"LLM visualization response: {response.content.strip()}")
            
            # Clean up markdown code blocks if present
            response_content = response.content.strip()
            if response_content.startswith('```json'):
                response_content = response_content.replace('```json', '').replace('```', '').strip()
            elif response_content.startswith('```'):
                response_content = response_content.replace('```', '').strip()
            
            func_call = json.loads(response_content)
            logger.info(f"Parsed function call: {func_call}")
            
            if func_call["function_name"] == "none":
                logger.info("No visualization selected by LLM")
                return None
                
            func_map = {
                "create_histogram": create_histogram,
                "create_line_chart": create_line_chart,
                "create_pie_chart": create_pie_chart,
                "create_bar_chart": create_bar_chart,
                "create_scatter_plot": create_scatter_plot,
                "create_box_plot": create_box_plot,
                "create_violin_plot": create_violin_plot,
                "create_heatmap": create_heatmap,
                "create_density_contour": create_density_contour,
                "create_density_heatmap": create_density_heatmap,
                "create_area_chart": create_area_chart,
                "create_funnel_chart": create_funnel_chart,
                "create_timeline_chart": create_timeline_chart,
                "create_sunburst_chart": create_sunburst_chart,
                "create_treemap_chart": create_treemap_chart,
                "create_icicle_chart": create_icicle_chart,
                "create_parallel_coordinates": create_parallel_coordinates,
                "create_parallel_categories": create_parallel_categories,
                "create_choropleth": create_choropleth,
                "create_scatter_geo": create_scatter_geo
            }
            
            func = func_map.get(func_call["function_name"], create_bar_chart)  # Fallback to bar chart
            args = func_call["arguments"]
            # Use the actual data from the query, not the sample data
            args["data"] = data
            
            # Validate arguments
            for key, value in args.items():
                if key in ["x_column", "y_column", "values_column", "names_column", "color_column", "locations_column", "lat_column", "lon_column", "size_column", "z_column", "x_start", "x_end", "y_column"]:
                    if isinstance(value, str) and value not in data.columns:
                        logger.warning(f"Invalid column {value} for {func.__name__}. Available columns: {list(data.columns)}")
                        return None
                elif key in ["path_columns", "dimensions"]:
                    if isinstance(value, list) and not all(col in data.columns for col in value):
                        logger.warning(f"Invalid columns {value} for {func.__name__}. Available columns: {list(data.columns)}")
                        return None
            
            logger.info(f"Creating visualization with function: {func_call['function_name']}")
            logger.info(f"Arguments: {args}")
            
            # Test if we can create a simple visualization first
            if func_call["function_name"] not in func_map:
                logger.warning(f"Unknown function: {func_call['function_name']}, using default bar chart")
                func = create_bar_chart
            else:
                func = func_map[func_call["function_name"]]
            
            fig = func(**args)
            logger.info("Visualization created successfully")
            return fig
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.info(f"LLM Response was: {response.content.strip()}")
            return self._create_fallback_visualization(data, user_question)
            
        except KeyError as e:
            logger.error(f"Missing key in LLM response: {e}")
            return self._create_fallback_visualization(data, user_question)
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return self._create_fallback_visualization(data, user_question)
    
    def _create_fallback_visualization(self, data: pd.DataFrame, user_question: str) -> Optional[go.Figure]:
        """Create a fallback visualization when LLM fails"""
        try:
            logger.info("Creating fallback visualization")
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            # Strategy 1: Bar chart for categorical vs numeric
            if len(numeric_cols) > 0 and len(categorical_cols) > 0 and len(data) <= 50:
                logger.info("Creating fallback bar chart")
                return create_bar_chart(
                    data=data,
                    x_column=categorical_cols[0],
                    y_column=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                )
            
            # Strategy 2: Histogram for single numeric column
            elif len(numeric_cols) > 0:
                logger.info("Creating fallback histogram")
                return create_histogram(
                    data=data,
                    x_column=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
            
            # Strategy 3: Pie chart if we have counts
            elif len(categorical_cols) > 0 and len(data) <= 20:
                logger.info("Creating fallback pie chart")
                # Count occurrences of the first categorical column
                counts = data[categorical_cols[0]].value_counts().reset_index()
                counts.columns = [categorical_cols[0], 'count']
                return create_pie_chart(
                    data=counts,
                    values_column='count',
                    names_column=categorical_cols[0],
                    title=f"Distribution of {categorical_cols[0]}"
                )
            
            logger.info("No suitable fallback visualization could be created")
            return None
            
        except Exception as fallback_error:
            logger.error(f"Fallback visualization also failed: {fallback_error}")
            return None
    
    def execute_query_safely(self, sql_query: str) -> Tuple[bool, pd.DataFrame, str]:
        """Execute SQL query with safety checks"""
        try:
            # Clean the query by removing comments and extra whitespace
            sql_lines = []
            for line in sql_query.split('\n'):
                # Remove SQL comments (lines starting with --)
                line = line.split('--')[0].strip()
                if line:  # Only add non-empty lines
                    sql_lines.append(line)
            
            cleaned_query = ' '.join(sql_lines).strip()
            sql_lower = cleaned_query.lower().strip()
            
            # Check for dangerous keywords
            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'truncate', 'create']
            for keyword in dangerous_keywords:
                if f' {keyword} ' in f' {sql_lower} ' or sql_lower.startswith(f'{keyword} '):
                    return False, pd.DataFrame(), f"Query contains potentially dangerous keyword: {keyword}"
            
            # Allow SELECT queries and CTEs (WITH clauses)
            allowed_starters = ['select', 'with', '/*']
            if not any(sql_lower.startswith(starter) for starter in allowed_starters):
                return False, pd.DataFrame(), "Only SELECT queries and CTEs (WITH clauses) are allowed"
            
            # Execute the original query (with comments, as they're safe)
            result_df = self.db_manager.execute_query(sql_query)
            
            if len(result_df) > 1000:
                original_length = len(result_df)
                result_df = result_df.head(1000)
                warning_msg = f"Results limited to 1000 rows (original query returned {original_length} rows)"
                return True, result_df, warning_msg
                
            return True, result_df, "Query executed successfully"
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error executing SQL query: {e}")
            
            # Provide specific guidance for common errors
            if "UNION types" in error_str and "cannot be matched" in error_str:
                return False, pd.DataFrame(), ("UNION type mismatch error. ALL columns must be cast in UNION operations. "
                                             "Required casting: CAST(state AS TEXT), CAST(district AS TEXT), CAST(rainfall_mm_total AS NUMERIC). "
                                             "The 'district' column is double precision in 2012-2013 but text in other tables.")
            elif "column" in error_str.lower() and "does not exist" in error_str.lower():
                if "perhaps you meant to reference" in error_str.lower():
                    return False, pd.DataFrame(), f"SQL logic error: Wrong table alias used. {error_str}"
                else:
                    return False, pd.DataFrame(), f"Column does not exist. Check the schema above for available columns. Error: {error_str}"
            else:
                return False, pd.DataFrame(), f"Query execution error: {error_str}"
    
    def generate_natural_response(self, user_question: str, query_results: pd.DataFrame, sql_query: str) -> str:
        """Generate natural language response based on query results"""
        if query_results.empty:
            return "No data found matching your query."
        results_summary = f"Query returned {len(query_results)} rows.\n\n"
        if len(query_results) <= 10:
            results_summary += "Complete results:\n"
            results_summary += query_results.to_string(index=False)
        else:
            results_summary += "Sample results (first 5 rows):\n"
            results_summary += query_results.head().to_string(index=False)
            results_summary += f"\n... and {len(query_results) - 5} more rows"
        numeric_cols = query_results.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            results_summary += "\n\nNumeric Summary:\n"
            for col in numeric_cols:
                if not query_results[col].isna().all():
                    mean_val = query_results[col].mean()
                    max_val = query_results[col].max()
                    min_val = query_results[col].min()
                    results_summary += f"- {col}: Mean={mean_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f}\n"
        response_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "results"],
            template="""
You are an expert groundwater resource analyst. Based on the user's question and the query results, provide a comprehensive and informative response.
User Question: {question}
SQL Query Used: {sql_query}
Query Results: {results}
Instructions:
1. Provide a clear, comprehensive answer to the user's question
2. Interpret the data in the context of groundwater resource management
3. Highlight key insights, trends, or patterns in the data
4. Explain technical terms when necessary
5. Use proper units (ham for hectare-meters, ha for hectares, mm for millimeters)
6. If relevant, mention the significance for groundwater management
7. Keep the response informative but accessible
8. Stick to the facts in the data, avoid speculation, and do not make assumptions beyond the data provided.
0. Give answered backed by data
10. Dont mention SQL anywhere in the response
Response:
"""
        )
        try:
            chain = response_prompt | self.llm
            response = chain.invoke({
                "question": user_question,
                "sql_query": sql_query,
                "results": results_summary
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            return f"I found the data you requested, but encountered an error generating the response. Here's a summary: {results_summary}"
    
    def process_user_query(self, user_question: str) -> Dict:
        """Main method to process user query end-to-end"""
        result = {
            'success': False,
            'sql_query': '',
            'data': pd.DataFrame(),
            'response': '',
            'error': '',
            'visualization': None
        }
        try:
            logger.info(f"Processing user question: {user_question}")
            sql_query = self.generate_sql_query(user_question)
            if not sql_query:
                result['error'] = "Failed to generate SQL query from your question"
                return result
            result['sql_query'] = sql_query
            success, data, message = self.execute_query_safely(sql_query)
            if not success:
                result['error'] = message
                return result
            result['data'] = data
            result['visualization'] = self.generate_visualization(user_question, data)
            natural_response = self.generate_natural_response(user_question, data, sql_query)
            result['response'] = natural_response
            result['success'] = True
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            result['error'] = f"An error occurred while processing your query: {str(e)}"
            return result

def main():
    """Test the query processor"""
    try:
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        test_questions = [
            "What are the top 5 states with highest groundwater recharge in 2024-2025?",
            "Show me the groundwater extraction data for Andhra Pradesh",
            "Which districts have the highest rainfall in 2023-2024?",
            "Plot groundwater recharge trends for Maharashtra",
            "Show recharge hierarchy by state and district",
            "Visualize rainfall vs. recharge in Karnataka"
        ]
        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("=" * 50)
            result = query_processor.process_user_query(question)
            if result['success']:
                print(f"SQL Query: {result['sql_query']}")
                print(f"Response: {result['response']}")
                if result['visualization']:
                    print(f"Visualization generated: {result['visualization'].layout.title.text}")
            else:
                print(f"Error: {result['error']}")
        db_manager.close_connection()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()