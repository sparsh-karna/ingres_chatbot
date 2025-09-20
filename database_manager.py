"""
Database setup and data loading module for INGRES ChatBot
Handles PostgreSQL database connection and CSV data ingestion
"""

import os
import re
import json
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.session = None
        self.metadata = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            # Get database configuration from environment variables
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'ingres_db')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', '')
            
            # Alternative: use DATABASE_URL if provided
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                self.engine = create_engine(database_url)
            else:
                database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                self.engine = create_engine(database_url)
            
            # Create session
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.metadata = MetaData()
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # This would need to be run with superuser privileges
            # For now, assume the database already exists
            pass
        except Exception as e:
            logger.error(f"Error creating database: {e}")
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names in the database"""
        try:
            inspector = self.engine.dialect.get_table_names(self.engine.connect())
            return inspector
        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> Dict:
        """Get schema information for a specific table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position;
                """))
                
                columns = []
                for row in result:
                    columns.append({
                        'column_name': row[0],
                        'data_type': row[1],
                        'is_nullable': row[2]
                    })
                
                return {
                    'table_name': table_name,
                    'columns': columns
                }
        except Exception as e:
            logger.error(f"Error getting table schema for {table_name}: {e}")
            return {}
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close database connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()


class DataLoader:
    """Handles loading CSV data into PostgreSQL database"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.csv_output_dir = "datasets/csv_output"
    
    def normalize_column_name(self, name):
        """Normalize column names for database compatibility"""
        # Convert to lowercase
        name = str(name).lower()
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-z0-9_]', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f"col_{name}"
        
        # For very long names, try to preserve uniqueness by keeping important parts
        if len(name) > 60:
            # Split by underscores and try to keep meaningful parts
            parts = name.split('_')
            
            # Keep the first few words and the last few words to preserve uniqueness
            if len(parts) > 6:
                # Keep first 4 parts and last 2 parts with ellipsis indicator
                shortened_parts = parts[:4] + ['abbrev'] + parts[-2:]
                name = '_'.join(shortened_parts)
            
            # If still too long, truncate but keep the end for uniqueness
            if len(name) > 60:
                name = name[:50] + '_' + name[-9:]  # Keep last 9 chars for uniqueness
        
        return name or "unnamed_column"
    
    def make_unique_column_names(self, columns):
        """Ensure all column names are unique by adding suffixes to duplicates"""
        seen = {}
        unique_columns = []
        
        for col in columns:
            original_col = col
            counter = 1
            
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            
            seen[col] = True
            unique_columns.append(col)
        
        return unique_columns
    
    def get_csv_files(self) -> List[str]:
        """Get list of CSV files in the csv_output directory"""
        csv_files = []
        if os.path.exists(self.csv_output_dir):
            for file in os.listdir(self.csv_output_dir):
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(self.csv_output_dir, file))
        return csv_files
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer appropriate PostgreSQL column types from DataFrame"""
        type_mapping = {}
        
        for col in df.columns:
            # Check if column contains only numeric data
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # If all values (excluding NaN) are numeric
                if not numeric_series.dropna().empty and len(numeric_series.dropna()) > 0:
                    # Check if integers
                    if all(numeric_series.dropna() == numeric_series.dropna().astype(int)):
                        type_mapping[col] = 'INTEGER'
                    else:
                        type_mapping[col] = 'FLOAT'
                else:
                    type_mapping[col] = 'TEXT'
            except:
                type_mapping[col] = 'TEXT'
        
        return type_mapping
    
    def create_table_from_csv(self, csv_file: str) -> bool:
        """Create table and load data from CSV file"""
        try:
            # Extract year from filename
            filename = os.path.basename(csv_file)
            year = filename.replace('.csv', '').replace('-', '_')
            table_name = f"groundwater_data_{year}"
            
            # Read CSV file
            logger.info(f"Processing CSV file: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Normalize column names
            normalized_columns = [self.normalize_column_name(col) for col in df.columns]
            # Make sure all column names are unique
            unique_columns = self.make_unique_column_names(normalized_columns)
            
            # Create column mapping
            column_mapping = dict(zip(df.columns, unique_columns))
            
            df = df.rename(columns=column_mapping)
            
            # Replace NaN values with None for database compatibility
            df = df.where(pd.notnull(df), None)
            
            # Drop the table if it exists
            with self.db_manager.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
            
            # Create and populate table
            df.to_sql(
                table_name, 
                self.db_manager.engine, 
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Successfully created table: {table_name} with {len(df)} rows")
            
            # Store column mapping for reference
            mapping_file = f"column_mappings_{year}.json"
            with open(mapping_file, 'w') as f:
                json.dump(column_mapping, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_file}: {e}")
            return False
    
    def load_all_csv_files(self):
        """Load all CSV files into the database"""
        csv_files = self.get_csv_files()
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.csv_output_dir}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        success_count = 0
        for csv_file in csv_files:
            if self.create_table_from_csv(csv_file):
                success_count += 1
        
        logger.info(f"Successfully loaded {success_count}/{len(csv_files)} CSV files")
    
    def get_database_summary(self) -> Dict:
        """Get summary of all tables in the database"""
        tables = {}
        
        try:
            with self.db_manager.engine.connect() as conn:
                # Get all table names
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'groundwater_data_%'
                """))
                
                table_names = [row[0] for row in result]
                
                for table_name in table_names:
                    # Get table info
                    schema = self.db_manager.get_table_schema(table_name)
                    
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.fetchone()[0]
                    
                    tables[table_name] = {
                        'schema': schema,
                        'row_count': row_count
                    }
        
        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
        
        return tables


def main():
    """Main function to load CSV data into database"""
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize data loader
        data_loader = DataLoader(db_manager)
        
        # Load all CSV files
        data_loader.load_all_csv_files()
        
        # Get and print database summary
        summary = data_loader.get_database_summary()
        print("\n=== Database Summary ===")
        for table_name, info in summary.items():
            print(f"Table: {table_name}")
            print(f"  Rows: {info['row_count']}")
            print(f"  Columns: {len(info['schema']['columns'])}")
            print()
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()