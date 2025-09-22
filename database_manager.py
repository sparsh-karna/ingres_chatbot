"""
Database setup and data loading module for INGRES ChatBot
Handles PostgreSQL database connection and CSV data ingestion
Also manages MongoDB for chat history storage
"""

import os
import re
import json
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import logging
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    """Manages database connections and operations for both PostgreSQL and MongoDB"""
    
    def __init__(self):
        # PostgreSQL attributes
        self.engine = None
        self.session = None
        self.metadata = None
        
        # MongoDB attributes
        self.mongo_client = None
        self.mongo_db = None
        self.chat_sessions_collection = None
        self.chat_messages_collection = None
        
        # Initialize connections
        self._initialize_connection()
        self._initialize_mongodb()
    
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
            logger.error(f"Failed to connect to PostgreSQL database: {e}")
            raise
    
    def _initialize_mongodb(self):
        """Initialize MongoDB connection and setup collections"""
        try:
            # Get MongoDB configuration from environment variables
            # Primary method: use MONGODB_URI (for Atlas/Cloud connections)
            mongodb_uri = os.getenv('MONGODB_URI')
            mongo_db_name = os.getenv('MONGODB_DATABASE_NAME', 'ingres_chatbot')
            
            if mongodb_uri:
                # Use provided URI (Atlas/Cloud connection)
                mongodb_url = mongodb_uri
            else:
                # Fallback: use individual components for local MongoDB
                mongo_host = os.getenv('MONGO_HOST', 'localhost')
                mongo_port = int(os.getenv('MONGO_PORT', '27017'))
                mongo_user = os.getenv('MONGO_USER', '')
                mongo_password = os.getenv('MONGO_PASSWORD', '')
                
                # Create MongoDB connection string
                if mongo_user and mongo_password:
                    mongodb_url = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
                else:
                    mongodb_url = f"mongodb://{mongo_host}:{mongo_port}/"
            
            # Connect to MongoDB
            self.mongo_client = MongoClient(mongodb_url)
            
            # Test the connection
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            
            # Get or create database
            self.mongo_db = self.mongo_client[mongo_db_name]
            logger.info(f"Connected to MongoDB database: {mongo_db_name}")
            
            # Setup collections
            self._setup_chat_collections()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("MongoDB functionality will be disabled. Chat history will be stored in memory only.")
            self.mongo_client = None
            self.mongo_db = None
    
    def _setup_chat_collections(self):
        """Setup MongoDB collections for chat functionality"""
        if self.mongo_db is None:
            return
        
        try:
            # Chat Sessions collection
            self.chat_sessions_collection = self.mongo_db['chat_sessions']
            
            # Chat Messages collection  
            self.chat_messages_collection = self.mongo_db['chat_messages']
            
            # Create indexes for better performance
            self.chat_sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
            self.chat_sessions_collection.create_index([("created_at", DESCENDING)])
            self.chat_sessions_collection.create_index([("last_activity", DESCENDING)])
            
            self.chat_messages_collection.create_index([("session_id", ASCENDING)])
            self.chat_messages_collection.create_index([("timestamp", DESCENDING)])
            self.chat_messages_collection.create_index([("session_id", ASCENDING), ("timestamp", DESCENDING)])
            
            logger.info("MongoDB chat collections setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up MongoDB collections: {e}")
            self.chat_sessions_collection = None
            self.chat_messages_collection = None
    
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
        
    def is_mongodb_available(self) -> bool:
        """Check if MongoDB is available"""
        return self.mongo_client is not None and self.mongo_db is not None
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                # Use text() to wrap the query for better SQLAlchemy compatibility
                result = conn.execute(text(query))
                
                # Convert result to DataFrame manually
                if result.returns_rows:
                    # Get column names
                    columns = list(result.keys())
                    # Get all rows
                    rows = result.fetchall()
                    
                    # Convert to DataFrame
                    if rows:
                        # Convert Row objects to lists/tuples for DataFrame creation
                        data = [list(row) for row in rows]
                        df = pd.DataFrame(data, columns=columns)
                    else:
                        df = pd.DataFrame(columns=columns)
                    
                    logger.info(f"Query executed successfully, returned {len(df)} rows")
                    return df
                else:
                    logger.info("Query executed but returned no rows")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    # ============ MongoDB Chat Management Methods ============
    
    def create_chat_session(self, session_id: str = None, metadata: Dict = None) -> str:
        """Create a new chat session in MongoDB"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot create chat session")
            return session_id or str(uuid.uuid4())
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session_doc = {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "message_count": 0,
                "metadata": metadata or {},
                "is_active": True
            }
            
            # Insert or update session
            self.chat_sessions_collection.replace_one(
                {"session_id": session_id}, 
                session_doc, 
                upsert=True
            )
            
            logger.info(f"Created chat session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return session_id or str(uuid.uuid4())
    
    def add_chat_message(self, session_id: str, role: str, content: str, 
                        metadata: Dict = None, sql_query: str = None, 
                        csv_data: str = None) -> bool:
        """Add a message to chat session in MongoDB"""
        if self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot store chat message")
            return False
        
        try:
            message_doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "sql_query": sql_query,
                "csv_data": csv_data
            }
            
            # Insert message
            result = self.chat_messages_collection.insert_one(message_doc)
            
            # Update session last activity and message count
            if self.chat_sessions_collection is not None:
                self.chat_sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {"last_activity": datetime.now(timezone.utc)},
                        "$inc": {"message_count": 1}
                    }
                )
            
            logger.info(f"Added chat message to session {session_id}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: int = 50, 
                        include_sql: bool = False) -> List[Dict]:
        """Get chat history for a session from MongoDB"""
        if self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot retrieve chat history")
            return []
        
        try:
            # Build projection
            projection = {
                "role": 1,
                "content": 1,
                "timestamp": 1,
                "metadata": 1,
                "_id": 0
            }
            
            if include_sql:
                projection["sql_query"] = 1
                projection["csv_data"] = 1
            
            # Get messages for session
            cursor = self.chat_messages_collection.find(
                {"session_id": session_id},
                projection
            ).sort("timestamp", ASCENDING).limit(limit)
            
            messages = list(cursor)
            
            # Convert datetime objects to ISO strings
            for message in messages:
                if isinstance(message.get('timestamp'), datetime):
                    message['timestamp'] = message['timestamp'].isoformat()
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []
    
    def get_recent_context(self, session_id: str, max_messages: int = 6) -> List[Dict]:
        """Get recent messages for context (last 3 exchanges = 6 messages)"""
        if self.chat_messages_collection is None:
            return []
        
        try:
            cursor = self.chat_messages_collection.find(
                {"session_id": session_id},
                {
                    "role": 1,
                    "content": 1,
                    "timestamp": 1,
                    "sql_query": 1,
                    "_id": 0
                }
            ).sort("timestamp", DESCENDING).limit(max_messages)
            
            messages = list(cursor)
            messages.reverse()  # Return in chronological order
            
            # Convert datetime objects to ISO strings
            for message in messages:
                if isinstance(message.get('timestamp'), datetime):
                    message['timestamp'] = message['timestamp'].isoformat()
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving recent context: {e}")
            return []
    
    def get_all_chat_sessions(self, limit: int = 100, active_only: bool = True) -> List[Dict]:
        """Get all chat sessions from MongoDB"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot retrieve chat sessions")
            return []
        
        try:
            query = {}
            if active_only:
                query["is_active"] = True
            
            cursor = self.chat_sessions_collection.find(
                query,
                {
                    "session_id": 1,
                    "created_at": 1,
                    "last_activity": 1,
                    "message_count": 1,
                    "metadata": 1,
                    "_id": 0
                }
            ).sort("last_activity", DESCENDING).limit(limit)
            
            sessions = list(cursor)
            
            # Convert datetime objects to ISO strings
            for session in sessions:
                for field in ['created_at', 'last_activity']:
                    if isinstance(session.get(field), datetime):
                        session[field] = session[field].isoformat()
            
            logger.info(f"Retrieved {len(sessions)} chat sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving chat sessions: {e}")
            return []
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        if self.chat_sessions_collection is None or self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot delete chat session")
            return False
        
        try:
            # Delete messages first
            message_result = self.chat_messages_collection.delete_many({"session_id": session_id})
            
            # Delete session
            session_result = self.chat_sessions_collection.delete_one({"session_id": session_id})
            
            logger.info(f"Deleted chat session {session_id}: {message_result.deleted_count} messages, {session_result.deleted_count} session")
            return session_result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False
    
    def deactivate_chat_session(self, session_id: str) -> bool:
        """Mark a chat session as inactive instead of deleting it"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot deactivate chat session")
            return False
        
        try:
            result = self.chat_sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"is_active": False, "deactivated_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"Deactivated chat session: {session_id}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error deactivating chat session: {e}")
            return False
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a chat session"""
        if self.chat_sessions_collection is None or self.chat_messages_collection is None:
            return {}
        
        try:
            # Get session info
            session = self.chat_sessions_collection.find_one({"session_id": session_id})
            if not session:
                return {}
            
            # Get message statistics
            message_stats = list(self.chat_messages_collection.aggregate([
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$role",
                    "count": {"$sum": 1}
                }}
            ]))
            
            stats = {
                "session_id": session_id,
                "created_at": session['created_at'].isoformat() if isinstance(session.get('created_at'), datetime) else session.get('created_at'),
                "last_activity": session['last_activity'].isoformat() if isinstance(session.get('last_activity'), datetime) else session.get('last_activity'),
                "message_count": session.get('message_count', 0),
                "is_active": session.get('is_active', False),
                "message_breakdown": {stat['_id']: stat['count'] for stat in message_stats}
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def is_mongodb_available(self) -> bool:
        """Check if MongoDB is available"""
        return self.mongo_client is not None and self.mongo_db is not None
    
    def close_connection(self):
        """Close database connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        
        # Also close MongoDB connection
        if self.mongo_client:
            self.mongo_client.close()


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
    
    # ============ MongoDB Chat Management Methods ============
    
    def create_chat_session(self, session_id: str = None, metadata: Dict = None) -> str:
        """Create a new chat session in MongoDB"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot create chat session")
            return session_id or str(uuid.uuid4())
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session_doc = {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "message_count": 0,
                "metadata": metadata or {},
                "is_active": True
            }
            
            # Insert or update session
            self.chat_sessions_collection.replace_one(
                {"session_id": session_id}, 
                session_doc, 
                upsert=True
            )
            
            logger.info(f"Created chat session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return session_id or str(uuid.uuid4())
    
    def add_chat_message(self, session_id: str, role: str, content: str, 
                        metadata: Dict = None, sql_query: str = None, 
                        csv_data: str = None) -> bool:
        """Add a message to chat session in MongoDB"""
        if self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot store chat message")
            return False
        
        try:
            message_doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "sql_query": sql_query,
                "csv_data": csv_data
            }
            
            # Insert message
            result = self.chat_messages_collection.insert_one(message_doc)
            
            # Update session last activity and message count
            if self.chat_sessions_collection is not None:
                self.chat_sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {"last_activity": datetime.now(timezone.utc)},
                        "$inc": {"message_count": 1}
                    }
                )
            
            logger.info(f"Added chat message to session {session_id}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: int = 50, 
                        include_sql: bool = False) -> List[Dict]:
        """Get chat history for a session from MongoDB"""
        if self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot retrieve chat history")
            return []
        
        try:
            # Build projection
            projection = {
                "role": 1,
                "content": 1,
                "timestamp": 1,
                "metadata": 1,
                "_id": 0
            }
            
            if include_sql:
                projection["sql_query"] = 1
                projection["csv_data"] = 1
            
            # Get messages for session
            cursor = self.chat_messages_collection.find(
                {"session_id": session_id},
                projection
            ).sort("timestamp", ASCENDING).limit(limit)
            
            messages = list(cursor)
            
            # Convert datetime objects to ISO strings
            for message in messages:
                if isinstance(message.get('timestamp'), datetime):
                    message['timestamp'] = message['timestamp'].isoformat()
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []
    
    def get_recent_context(self, session_id: str, max_messages: int = 6) -> List[Dict]:
        """Get recent messages for context (last 3 exchanges = 6 messages)"""
        if self.chat_messages_collection is None:
            return []
        
        try:
            cursor = self.chat_messages_collection.find(
                {"session_id": session_id},
                {
                    "role": 1,
                    "content": 1,
                    "timestamp": 1,
                    "sql_query": 1,
                    "_id": 0
                }
            ).sort("timestamp", DESCENDING).limit(max_messages)
            
            messages = list(cursor)
            messages.reverse()  # Return in chronological order
            
            # Convert datetime objects to ISO strings
            for message in messages:
                if isinstance(message.get('timestamp'), datetime):
                    message['timestamp'] = message['timestamp'].isoformat()
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving recent context: {e}")
            return []
    
    def get_all_chat_sessions(self, limit: int = 100, active_only: bool = True) -> List[Dict]:
        """Get all chat sessions from MongoDB"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot retrieve chat sessions")
            return []
        
        try:
            query = {}
            if active_only:
                query["is_active"] = True
            
            cursor = self.chat_sessions_collection.find(
                query,
                {
                    "session_id": 1,
                    "created_at": 1,
                    "last_activity": 1,
                    "message_count": 1,
                    "metadata": 1,
                    "_id": 0
                }
            ).sort("last_activity", DESCENDING).limit(limit)
            
            sessions = list(cursor)
            
            # Convert datetime objects to ISO strings
            for session in sessions:
                for field in ['created_at', 'last_activity']:
                    if isinstance(session.get(field), datetime):
                        session[field] = session[field].isoformat()
            
            logger.info(f"Retrieved {len(sessions)} chat sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving chat sessions: {e}")
            return []
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        if self.chat_sessions_collection is None or self.chat_messages_collection is None:
            logger.warning("MongoDB not available, cannot delete chat session")
            return False
        
        try:
            # Delete messages first
            message_result = self.chat_messages_collection.delete_many({"session_id": session_id})
            
            # Delete session
            session_result = self.chat_sessions_collection.delete_one({"session_id": session_id})
            
            logger.info(f"Deleted chat session {session_id}: {message_result.deleted_count} messages, {session_result.deleted_count} session")
            return session_result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False
    
    def deactivate_chat_session(self, session_id: str) -> bool:
        """Mark a chat session as inactive instead of deleting it"""
        if self.chat_sessions_collection is None:
            logger.warning("MongoDB not available, cannot deactivate chat session")
            return False
        
        try:
            result = self.chat_sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"is_active": False, "deactivated_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"Deactivated chat session: {session_id}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error deactivating chat session: {e}")
            return False
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a chat session"""
        if self.chat_sessions_collection is None or self.chat_messages_collection is None:
            return {}
        
        try:
            # Get session info
            session = self.chat_sessions_collection.find_one({"session_id": session_id})
            if not session:
                return {}
            
            # Get message statistics
            message_stats = list(self.chat_messages_collection.aggregate([
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$role",
                    "count": {"$sum": 1}
                }}
            ]))
            
            stats = {
                "session_id": session_id,
                "created_at": session['created_at'].isoformat() if isinstance(session.get('created_at'), datetime) else session.get('created_at'),
                "last_activity": session['last_activity'].isoformat() if isinstance(session.get('last_activity'), datetime) else session.get('last_activity'),
                "message_count": session.get('message_count', 0),
                "is_active": session.get('is_active', False),
                "message_breakdown": {stat['_id']: stat['count'] for stat in message_stats}
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def close_connections(self):
        """Close both PostgreSQL and MongoDB connections"""
        try:
            # Close PostgreSQL
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            
            # Close MongoDB
            if self.mongo_client:
                self.mongo_client.close()
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def is_mongodb_available(self) -> bool:
        """Check if MongoDB is available"""
        return self.mongo_client is not None and self.mongo_db is not None
    
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