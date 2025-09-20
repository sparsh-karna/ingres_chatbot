"""
AI Query Processing Module for INGRES ChatBot
Handles natural language to SQL conversion using Google Gemini
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from sqlalchemy import text
from database_manager import DatabaseManager
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes natural language queries and converts them to SQL"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.llm = None
        self._initialize_llm()
        self.database_schema = self._get_database_schema()
    
    def _initialize_llm(self):
        """Initialize Google Gemini LLM"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.1
            )
            logger.info("Google Gemini LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _get_database_schema(self) -> Dict:
        """Get comprehensive database schema information"""
        schema_info = {}
        
        try:
            with self.db_manager.engine.connect() as conn:
                # Get all groundwater data tables
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
    
    def _create_schema_context(self) -> str:
        """Create a formatted schema context for the LLM"""
        schema_context = "Database Schema Information:\n\n"
        
        for table_name, schema in self.database_schema.items():
            # Extract year from table name for better context
            year = table_name.replace('groundwater_data_', '').replace('_', '-')
            
            schema_context += f"Table: {table_name} (Year: {year})\n"
            schema_context += "Columns:\n"
            
            for col in schema['columns']:
                col_name = col['column_name']
                col_type = col['data_type']
                nullable = col['is_nullable']
                schema_context += f"  - {col_name} ({col_type})\n"
            
            schema_context += "\n"
        
        # Add domain knowledge context
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
        """Generate SQL query from natural language question"""
        
        schema_context = self._create_schema_context()
        
        sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
You are an expert SQL query generator for a groundwater resource database.

{schema}

User Question: {question}

Instructions:
1. Generate a SQL query that answers the user's question
2. Use appropriate table names and column names from the schema
3. Include proper WHERE clauses, JOINs, and aggregations as needed
4. For time-based queries, consider which year's data to use
5. Return ONLY the SQL query, no explanations
6. Ensure the query is PostgreSQL compatible
7. Use proper column aliases for better readability
8. Limit results to reasonable numbers (use LIMIT if needed)

SQL Query:
"""
        )
        
        try:
            chain = sql_prompt | self.llm
            response = chain.invoke({
                "schema": schema_context,
                "question": user_question
            })
            
            # Clean up the response to extract only the SQL
            sql_query = response.content.strip()
            
            # Remove any markdown formatting
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            elif sql_query.startswith('```'):
                sql_query = sql_query.replace('```', '').strip()
            
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return ""
    
    def execute_query_safely(self, sql_query: str) -> Tuple[bool, pd.DataFrame, str]:
        """Execute SQL query with safety checks"""
        try:
            # Basic safety checks
            sql_lower = sql_query.lower().strip()
            
            # Block potentially dangerous operations
            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'truncate', 'create']
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    return False, pd.DataFrame(), f"Query contains potentially dangerous keyword: {keyword}"
            
            # Ensure it's a SELECT query
            if not sql_lower.startswith('select'):
                return False, pd.DataFrame(), "Only SELECT queries are allowed"
            
            # Execute the query
            result_df = self.db_manager.execute_query(sql_query)
            
            # Limit results if too large
            if len(result_df) > 1000:
                result_df = result_df.head(1000)
                warning_msg = f"Results limited to 1000 rows (original query returned {len(result_df)} rows)"
                return True, result_df, warning_msg
            
            return True, result_df, "Query executed successfully"
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return False, pd.DataFrame(), f"Query execution error: {str(e)}"
    
    def generate_natural_response(self, user_question: str, query_results: pd.DataFrame, sql_query: str) -> str:
        """Generate natural language response based on query results"""
        
        # Convert DataFrame to a readable format
        if query_results.empty:
            return "No data found matching your query."
        
        # Prepare results summary
        results_summary = f"Query returned {len(query_results)} rows.\n\n"
        
        # Add sample of the data
        if len(query_results) <= 10:
            results_summary += "Complete results:\n"
            results_summary += query_results.to_string(index=False)
        else:
            results_summary += "Sample results (first 5 rows):\n"
            results_summary += query_results.head().to_string(index=False)
            results_summary += f"\n... and {len(query_results) - 5} more rows"
        
        # Add basic statistics if numeric columns exist
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
            'error': ''
        }
        
        try:
            # Step 1: Generate SQL query
            logger.info(f"Processing user question: {user_question}")
            sql_query = self.generate_sql_query(user_question)
            
            if not sql_query:
                result['error'] = "Failed to generate SQL query from your question"
                return result
            
            result['sql_query'] = sql_query
            
            # Step 2: Execute SQL query
            success, data, message = self.execute_query_safely(sql_query)
            
            if not success:
                result['error'] = message
                return result
            
            result['data'] = data
            
            # Step 3: Generate natural language response
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
        # Initialize components
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        
        # Test queries
        test_questions = [
            "What are the top 5 states with highest groundwater recharge in 2024-2025?",
            "Show me the groundwater extraction data for Andhra Pradesh",
            "Which districts have the highest rainfall in 2023-2024?",
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("=" * 50)
            
            result = query_processor.process_user_query(question)
            
            if result['success']:
                print(f"SQL Query: {result['sql_query']}")
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result['error']}")
        
        db_manager.close_connection()
        
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()