"""
Demo script to test the INGRES ChatBot pipeline
This script demonstrates the basic functionality without requiring full setup
"""

import os
import pandas as pd
import json
from datetime import datetime

def demo_csv_analysis():
    """Analyze the CSV files to understand the data structure"""
    print("🔍 Analyzing CSV files in datasets/csv_output/")
    print("=" * 60)
    
    csv_dir = "datasets/csv_output"
    
    if not os.path.exists(csv_dir):
        print("❌ CSV directory not found!")
        return
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for csv_file in sorted(csv_files):
        print(f"\n📊 Analyzing: {csv_file}")
        file_path = os.path.join(csv_dir, csv_file)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Basic statistics
            print(f"  📈 Rows: {len(df):,}")
            print(f"  📋 Columns: {len(df.columns)}")
            
            # Show column types
            numeric_cols = df.select_dtypes(include=['number']).columns
            text_cols = df.select_dtypes(include=['object']).columns
            
            print(f"  🔢 Numeric columns: {len(numeric_cols)}")
            print(f"  📝 Text columns: {len(text_cols)}")
            
            # Show sample data
            print(f"  🔍 Sample data:")
            if not df.empty:
                # Show first few key columns
                key_cols = ['STATE', 'DISTRICT', 'ASSESSMENT UNIT']
                available_cols = [col for col in key_cols if col in df.columns]
                
                if available_cols:
                    sample_data = df[available_cols].head(3)
                    for idx, row in sample_data.iterrows():
                        print(f"    - {', '.join([f'{col}: {row[col]}' for col in available_cols])}")
                else:
                    print(f"    - First column: {df.iloc[0, 0] if len(df) > 0 else 'No data'}")
            
            # Show unique states/districts if available
            if 'STATE' in df.columns:
                unique_states = df['STATE'].nunique()
                print(f"  🏛️ Unique states: {unique_states}")
            
            if 'DISTRICT' in df.columns:
                unique_districts = df['DISTRICT'].nunique()
                print(f"  🏘️ Unique districts: {unique_districts}")
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")


def demo_sql_generation():
    """Demonstrate SQL generation logic (without actual LLM)"""
    print("\n🤖 Demonstrating SQL Generation Logic")
    print("=" * 60)
    
    # Sample questions and their expected SQL patterns
    sample_queries = [
        {
            "question": "What are the top 5 states with highest groundwater recharge in 2024-2025?",
            "expected_pattern": "SELECT STATE, SUM(groundwater_recharge) FROM groundwater_data_2024_2025 GROUP BY STATE ORDER BY SUM(groundwater_recharge) DESC LIMIT 5"
        },
        {
            "question": "Show me groundwater extraction data for Maharashtra",
            "expected_pattern": "SELECT * FROM groundwater_data_* WHERE STATE = 'MAHARASHTRA'"
        },
        {
            "question": "Which districts have the highest rainfall in 2023-2024?",
            "expected_pattern": "SELECT DISTRICT, rainfall_total FROM groundwater_data_2023_2024 ORDER BY rainfall_total DESC"
        }
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. Question: {query['question']}")
        print(f"   Expected SQL pattern: {query['expected_pattern']}")
        
        # Simple pattern matching logic (placeholder for actual LLM)
        if "top" in query['question'].lower() and "state" in query['question'].lower():
            print("   ✅ Pattern: Aggregation query with GROUP BY state")
        elif "maharashtra" in query['question'].lower():
            print("   ✅ Pattern: Filter query by specific state")
        elif "rainfall" in query['question'].lower():
            print("   ✅ Pattern: Sort query by rainfall column")


def demo_database_schema():
    """Show what the database schema would look like"""
    print("\n🗂️ Expected Database Schema")
    print("=" * 60)
    
    # Read a sample header file to show structure
    header_file = "datasets/csv_output/2024-2025_headers.json"
    
    if os.path.exists(header_file):
        try:
            with open(header_file, 'r') as f:
                headers = json.load(f)
            
            print(f"📊 Table: groundwater_data_2024_2025")
            print(f"   Total columns: {len(headers)}")
            print(f"   Sample columns:")
            
            # Group columns by category
            categories = {
                'Basic Info': ['S.No', 'STATE', 'DISTRICT', 'ASSESSMENT UNIT'],
                'Rainfall': [h for h in headers if 'Rainfall' in h][:3],
                'Groundwater Recharge': [h for h in headers if 'Ground Water Recharge' in h][:3],
                'Extraction': [h for h in headers if 'Extraction' in h][:3]
            }
            
            for category, cols in categories.items():
                if cols:
                    print(f"   {category}:")
                    for col in cols:
                        # Normalize column name for database
                        normalized = col.lower().replace(' ', '_').replace('(', '_').replace(')', '_')
                        normalized = normalized.replace('%', 'percent').replace('-', '_')
                        print(f"     - {col} → {normalized}")
        
        except Exception as e:
            print(f"❌ Error reading header file: {e}")
    else:
        print("❌ Header file not found")


def demo_response_generation():
    """Demonstrate response generation"""
    print("\n💬 Demo Response Generation")
    print("=" * 60)
    
    # Sample query result
    sample_data = {
        'STATE': ['MAHARASHTRA', 'UTTAR PRADESH', 'RAJASTHAN', 'MADHYA PRADESH', 'GUJARAT'],
        'total_recharge': [45000.5, 42300.8, 38900.2, 36500.1, 35200.9],
        'avg_rainfall': [850.2, 920.5, 650.1, 780.3, 720.8]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample Query: 'What are the top 5 states with highest groundwater recharge?'")
    print("\nSample Data Retrieved:")
    print(df.to_string(index=False))
    
    print("\nGenerated Response (simulated):")
    response = f"""
Based on the latest groundwater assessment data, here are the top 5 states with the highest groundwater recharge:

1. **Maharashtra**: {df.iloc[0]['total_recharge']:,.1f} hectare-meters of annual groundwater recharge
2. **Uttar Pradesh**: {df.iloc[1]['total_recharge']:,.1f} hectare-meters 
3. **Rajasthan**: {df.iloc[2]['total_recharge']:,.1f} hectare-meters
4. **Madhya Pradesh**: {df.iloc[3]['total_recharge']:,.1f} hectare-meters
5. **Gujarat**: {df.iloc[4]['total_recharge']:,.1f} hectare-meters

These states show strong groundwater recharge capacity, with Maharashtra leading at {df.iloc[0]['total_recharge']:,.1f} ham. 
The average rainfall in these states ranges from {df['avg_rainfall'].min():.1f}mm to {df['avg_rainfall'].max():.1f}mm, 
which correlates with their recharge potential.

This data is crucial for groundwater resource planning and sustainable water management policies.
"""
    print(response)


def main():
    """Run the demo"""
    print("🚀 INGRES ChatBot Pipeline Demo")
    print(f"📅 Demo run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Run demo components
        demo_csv_analysis()
        demo_database_schema()
        demo_sql_generation()
        demo_response_generation()
        
        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Set up PostgreSQL database")
        print("2. Configure .env file with database credentials and Google API key")
        print("3. Run 'python cli.py --setup' to load data")
        print("4. Run 'python cli.py --chat' for interactive testing")
        print("5. Run 'streamlit run streamlit_app.py' for web interface")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()