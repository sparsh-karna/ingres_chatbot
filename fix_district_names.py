#!/usr/bin/env python3
"""
Script to capitalize district names in all groundwater data tables
This ensures consistency with the LLM's expectation of uppercase district names
"""

import os
import sys
from database_manager import DatabaseManager
from sqlalchemy import text
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def capitalize_district_names():
    """Capitalize all district names in all groundwater data tables"""
    
    # Initialize database manager
    db = DatabaseManager()
    
    try:
        # Get all groundwater data tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE 'groundwater_data_%'
        ORDER BY table_name;
        """
        
        tables_result = db.execute_query(tables_query)
        table_names = tables_result['table_name'].tolist()
        
        logger.info(f"Found {len(table_names)} tables to process: {table_names}")
        
        total_updated = 0
        
        for table_name in table_names:
            logger.info(f"\nProcessing table: {table_name}")
            
            # First, check current district values
            check_query = f"""
            SELECT DISTINCT district, COUNT(*) as count
            FROM {table_name} 
            WHERE district IS NOT NULL 
            GROUP BY district 
            ORDER BY district;
            """
            
            current_districts = db.execute_query(check_query)
            logger.info(f"Found {len(current_districts)} unique districts in {table_name}")
            
            # Show sample of current values
            if not current_districts.empty:
                logger.info("Sample districts before update:")
                for idx, row in current_districts.head(10).iterrows():
                    logger.info(f"  '{row['district']}' ({row['count']} records)")
            
            # Update district names to uppercase
            update_query = f"""
            UPDATE {table_name} 
            SET district = UPPER(district) 
            WHERE district IS NOT NULL 
            AND district != UPPER(district);
            """
            
            # Execute update using raw SQL since this is a data modification
            try:
                with db.engine.connect() as conn:
                    result = conn.execute(text(update_query))
                    conn.commit()
                    updated_count = result.rowcount
                    total_updated += updated_count
                    logger.info(f"Updated {updated_count} records in {table_name}")
            except Exception as e:
                logger.error(f"Error updating {table_name}: {e}")
                continue
            
            # Verify the update
            verify_query = f"""
            SELECT DISTINCT district, COUNT(*) as count
            FROM {table_name} 
            WHERE district IS NOT NULL 
            GROUP BY district 
            ORDER BY district
            LIMIT 10;
            """
            
            updated_districts = db.execute_query(verify_query)
            logger.info("Sample districts after update:")
            for idx, row in updated_districts.head(10).iterrows():
                logger.info(f"  '{row['district']}' ({row['count']} records)")
        
        logger.info(f"\n✅ COMPLETED: Updated {total_updated} total records across {len(table_names)} tables")
        
        # Final verification - show some examples
        logger.info("\n🔍 FINAL VERIFICATION:")
        sample_query = f"""
        SELECT DISTINCT state, district 
        FROM {table_names[0]} 
        WHERE state = 'MAHARASHTRA' 
        ORDER BY district 
        LIMIT 10;
        """
        
        sample_result = db.execute_query(sample_query)
        logger.info(f"Sample districts from {table_names[0]}:")
        for idx, row in sample_result.iterrows():
            logger.info(f"  State: '{row['state']}', District: '{row['district']}'")
            
    except Exception as e:
        logger.error(f"Error in capitalize_district_names: {e}")
        raise
    finally:
        db.close_connection()

def test_beed_query():
    """Test the Beed query after capitalization"""
    db = DatabaseManager()
    
    try:
        logger.info("\n🧪 TESTING BEED QUERY:")
        
        test_query = """
        SELECT DISTINCT state, district 
        FROM groundwater_data_2016_2017 
        WHERE district = 'BEED'
        """
        
        result = db.execute_query(test_query)
        logger.info(f"Found {len(result)} records for BEED district:")
        print(result)
        
        # Also test the original problematic query
        complex_query = """
        SELECT '2016-2017' AS year, state, district, 
               SUM(rainfall_mm_total) AS total_rainfall,
               SUM(ground_water_extraction_for_all_uses_ha_m_total) AS total_extraction,
               SUM(ground_water_recharge_ham_total) AS total_recharge,
               SUM(total_geographical_area_ha_total) AS total_area 
        FROM groundwater_data_2016_2017 
        WHERE district = 'BEED' 
        GROUP BY state, district;
        """
        
        complex_result = db.execute_query(complex_query)
        logger.info(f"Complex query result for BEED: {len(complex_result)} rows")
        if not complex_result.empty:
            print(complex_result)
        
    except Exception as e:
        logger.error(f"Error in test_beed_query: {e}")
    finally:
        db.close_connection()

if __name__ == "__main__":
    print("🚀 Starting district name capitalization process...")
    print("This will update all district names to uppercase for consistency with LLM queries.")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        try:
            capitalize_district_names()
            test_beed_query()
            print("\n✅ Successfully completed district name capitalization!")
            print("The LLM should now be able to find 'BEED' district in queries.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    else:
        print("❌ Operation cancelled by user.")
        sys.exit(0)