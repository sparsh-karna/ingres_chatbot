"""
INGRES AI ChatBot - Streamlit Web Interface
Interactive web application for querying groundwater resource data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from sqlalchemy import text
from database_manager import DatabaseManager
from query_processor import QueryProcessor
import logging
from typing import Dict

# Configure page
st.set_page_config(
    page_title="INGRES AI ChatBot",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5282;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #3182ce;
        margin: 1rem 0;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = None

@st.cache_resource
def initialize_components():
    """Initialize database manager and query processor"""
    try:
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        return db_manager, query_processor
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None

def display_query_result(result: Dict):
    """Display query results with appropriate formatting"""
    if not result['success']:
        st.error(f"❌ Error: {result['error']}")
        return
    
    # Display SQL query in expandable section
    with st.expander("🔍 Generated SQL Query", expanded=False):
        st.markdown(f"```sql\n{result['sql_query']}\n```")
    
    # Display natural language response
    st.markdown("### 📊 Analysis")
    st.markdown(result['response'])
    
    # Display data table
    if not result['data'].empty:
        st.markdown("### 📋 Data Table")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(result['data']))
        with col2:
            st.metric("Total Columns", len(result['data'].columns))
        with col3:
            if result['data'].select_dtypes(include=['number']).empty:
                st.metric("Numeric Columns", 0)
            else:
                st.metric("Numeric Columns", len(result['data'].select_dtypes(include=['number']).columns))
        st.dataframe(result['data'], use_container_width=True, height=400)
        
        # Display visualization
        if result['visualization']:
            st.markdown("### 📈 Visualization")
            st.plotly_chart(result['visualization'], use_container_width=True)
        else:
            # Check if data has visualizable columns
            numeric_cols = result['data'].select_dtypes(include=['number']).columns
            categorical_cols = result['data'].select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 or len(categorical_cols) > 0:
                st.info("💡 **Tip**: Add words like 'plot', 'chart', 'visualize', 'show', or 'top' to your question to generate visualizations!")
                
                # Offer a manual visualization button
                if st.button("🎨 Create Visualization Anyway"):
                    st.rerun()
            else:
                st.info("ℹ️ No visualizable data columns found in the results.")
        
        # Download option
        csv = result['data'].to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"ingres_query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">💧 INGRES AI ChatBot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Intelligent Virtual Assistant for India Ground Water Resource Estimation System</p>', unsafe_allow_html=True)
    
    if st.session_state.db_manager is None or st.session_state.query_processor is None:
        with st.spinner("🔄 Initializing INGRES ChatBot..."):
            db_manager, query_processor = initialize_components()
            if db_manager and query_processor:
                st.session_state.db_manager = db_manager
                st.session_state.query_processor = query_processor
                st.success("✅ ChatBot initialized successfully!")
            else:
                st.error("❌ Failed to initialize ChatBot. Please check your configuration.")
                st.stop()
    
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("### 📊 Database Information")
        try:
            with st.session_state.db_manager.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name) as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'groundwater_data_%'
                    ORDER BY table_name
                """))
                for row in result:
                    table_name = row[0]
                    year = table_name.replace('groundwater_data_', '').replace('_', '-')
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.fetchone()[0]
                    st.write(f"**{year}**: {row_count:,} records")
        except Exception as e:
            st.error(f"Error loading database info: {e}")
        
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What are the top 10 districts with highest groundwater recharge in 2024-2025?",
            "Show me groundwater extraction data for Maharashtra",
            "Which states have critical groundwater extraction levels?",
            "Compare rainfall patterns between 2020 and 2024",
            "What is the average groundwater recharge in Rajasthan?",
            "Show me districts with over-exploited groundwater resources",
            "Plot groundwater recharge trends for Maharashtra",
            "Show recharge hierarchy by state and district",
            "Visualize rainfall vs. recharge in Karnataka"
        ]
        for i, question in enumerate(sample_questions):
            if st.button(f"📝 {question[:50]}...", key=f"sample_{i}"):
                st.session_state.current_question = question
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🤖 Ask Your Question")
        user_question = st.text_area(
            "Enter your question about groundwater resources:",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="e.g., What are the top 5 states with highest groundwater recharge in 2024-2025?"
        )
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            submit_button = st.button("🚀 Ask ChatBot", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            if st.button("📜 Query History", use_container_width=True):
                st.session_state.show_history = not st.session_state.get('show_history', False)
    
    with col2:
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        <div class="info-box">
        <strong>Tips for better queries:</strong><br>
        • Be specific about years (e.g., "2024-2025")<br>
        • Mention specific states or districts<br>
        • Ask about specific metrics like rainfall, recharge, extraction<br>
        • Use comparative language for trends<br>
        • Include 'plot', 'chart', or 'visualize' for visualizations<br>
        </div>
        """, unsafe_allow_html=True)
    
    if clear_button:
        st.session_state.current_question = ''
        st.rerun()
    
    if submit_button and user_question.strip():
        with st.spinner("🔍 Processing your question..."):
            try:
                result = st.session_state.query_processor.process_user_query(user_question)
                st.session_state.query_history.append({
                    'timestamp': datetime.now(),
                    'question': user_question,
                    'result': result
                })
                display_query_result(result)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
    
    if st.session_state.get('show_history', False) and st.session_state.query_history:
        st.markdown("### 📜 Query History")
        for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"🕒 {entry['timestamp'].strftime('%H:%M:%S')} - {entry['question'][:60]}..."):
                display_query_result(entry['result'])
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Developed by:** Your Team")
    with col2:
        st.markdown("**For:** Central Ground Water Board (CGWB)")
    with col3:
        st.markdown("**Powered by:** Google Gemini AI")

if __name__ == "__main__":
    main()