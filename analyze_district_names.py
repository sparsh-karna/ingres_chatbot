#!/usr/bin/env python3
"""
Alternative script to check current district name formats and suggest improvements
without modifying the database
"""

from database_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_district_names():
    """Analyze current district name formats across all tables"""
    db = DatabaseManager()
    
    try:
        # Get all tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE 'groundwater_data_%'
        ORDER BY table_name;
        """
        
        tables_result = db.execute_query(tables_query)
        table_names = tables_result['table_name'].tolist()
        
        print(f"📊 Analyzing district names in {len(table_names)} tables...")
        
        all_districts = set()
        case_issues = []
        
        for table_name in table_names:
            query = f"""
            SELECT DISTINCT state, district 
            FROM {table_name} 
            WHERE district IS NOT NULL 
            ORDER BY state, district;
            """
            
            result = db.execute_query(query)
            
            for _, row in result.iterrows():
                state = row['state']
                district = row['district']
                all_districts.add((state, district))
                
                # Check if district is not all uppercase
                if district != district.upper():
                    case_issues.append((table_name, state, district, district.upper()))
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"Total unique (state, district) combinations: {len(all_districts)}")
        print(f"Districts with case issues: {len(case_issues)}")
        
        if case_issues:
            print(f"\n🔍 SAMPLE CASE ISSUES (showing first 20):")
            for table, state, original, uppercase in case_issues[:20]:
                print(f"  Table: {table}")
                print(f"    '{state}' -> '{original}' should be '{uppercase}'")
        
        # Show specific examples for Maharashtra
        print(f"\n🏛️ MAHARASHTRA DISTRICTS:")
        maharashtra_districts = [d for s, d in all_districts if s == 'MAHARASHTRA']
        print(f"Found {len(maharashtra_districts)} districts in Maharashtra:")
        for district in sorted(maharashtra_districts)[:15]:  # Show first 15
            upper_version = district.upper()
            status = "✅ OK" if district == upper_version else f"❌ Should be {upper_version}"
            print(f"  '{district}' {status}")
        
        # Test specific queries
        print(f"\n🧪 TESTING QUERIES:")
        
        # Test case-sensitive query
        test_queries = [
            ("Case-sensitive 'BEED'", "SELECT COUNT(*) as count FROM groundwater_data_2016_2017 WHERE district = 'BEED'"),
            ("Case-insensitive 'BEED'", "SELECT COUNT(*) as count FROM groundwater_data_2016_2017 WHERE UPPER(district) = 'BEED'"),
            ("Case-sensitive 'Beed'", "SELECT COUNT(*) as count FROM groundwater_data_2016_2017 WHERE district = 'Beed'"),
            ("All Beed variants", "SELECT DISTINCT district FROM groundwater_data_2016_2017 WHERE UPPER(district) = 'BEED'")
        ]
        
        for description, query in test_queries:
            try:
                result = db.execute_query(query)
                print(f"  {description}: {result.iloc[0, 0] if not result.empty else 'No results'}")
            except Exception as e:
                print(f"  {description}: Error - {e}")
        
    except Exception as e:
        logger.error(f"Error in analyze_district_names: {e}")
    finally:
        db.close_connection()

if __name__ == "__main__":
    analyze_district_names()