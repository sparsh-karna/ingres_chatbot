#!/usr/bin/env python3
"""
MongoDB Setup and Verification Script for INGRES ChatBot
Checks if MongoDB is running and sets up the database if needed
"""

import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

def check_mongodb_connection():
    """Check if MongoDB is running and accessible"""
    load_dotenv()
    
    # Get MongoDB configuration
    mongo_host = os.getenv('MONGO_HOST', 'localhost')
    mongo_port = int(os.getenv('MONGO_PORT', '27017'))
    mongo_db_name = os.getenv('MONGO_DB_NAME', 'ingres_chat_db')
    mongo_user = os.getenv('MONGO_USER', '')
    mongo_password = os.getenv('MONGO_PASSWORD', '')
    
    # Create connection string
    if mongo_user and mongo_password:
        mongo_url = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
    else:
        mongo_url = f"mongodb://{mongo_host}:{mongo_port}/"
    
    mongodb_url = os.getenv('MONGODB_URL', mongo_url)
    
    print(f"Checking MongoDB connection at: {mongodb_url}")
    
    try:
        # Create client with a short timeout
        client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Get or create database
        db = client[mongo_db_name]
        print(f"✅ Database '{mongo_db_name}' is accessible")
        
        # Test collections creation
        collections = ['chat_sessions', 'chat_messages']
        for collection_name in collections:
            collection = db[collection_name]
            # Insert and remove a test document to ensure write permissions
            test_doc = {"_test": True}
            result = collection.insert_one(test_doc)
            collection.delete_one({"_id": result.inserted_id})
            print(f"✅ Collection '{collection_name}' is writable")
        
        client.close()
        return True
        
    except ConnectionFailure:
        print("❌ Failed to connect to MongoDB: Connection refused")
        print("Make sure MongoDB is running on the specified host and port")
        return False
        
    except ServerSelectionTimeoutError:
        print("❌ Failed to connect to MongoDB: Server selection timeout")
        print("Check if MongoDB is running and accessible")
        return False
        
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return False

def setup_mongodb_indexes():
    """Setup MongoDB indexes for better performance"""
    load_dotenv()
    
    mongo_host = os.getenv('MONGO_HOST', 'localhost')
    mongo_port = int(os.getenv('MONGO_PORT', '27017'))
    mongo_db_name = os.getenv('MONGO_DB_NAME', 'ingres_chat_db')
    mongo_user = os.getenv('MONGO_USER', '')
    mongo_password = os.getenv('MONGO_PASSWORD', '')
    
    if mongo_user and mongo_password:
        mongo_url = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
    else:
        mongo_url = f"mongodb://{mongo_host}:{mongo_port}/"
    
    mongodb_url = os.getenv('MONGODB_URL', mongo_url)
    
    try:
        client = MongoClient(mongodb_url)
        db = client[mongo_db_name]
        
        # Setup indexes for chat_sessions collection
        chat_sessions = db['chat_sessions']
        chat_sessions.create_index([("session_id", 1)], unique=True)
        chat_sessions.create_index([("created_at", -1)])
        chat_sessions.create_index([("last_activity", -1)])
        print("✅ Chat sessions indexes created")
        
        # Setup indexes for chat_messages collection
        chat_messages = db['chat_messages']
        chat_messages.create_index([("session_id", 1)])
        chat_messages.create_index([("timestamp", -1)])
        chat_messages.create_index([("session_id", 1), ("timestamp", -1)])
        print("✅ Chat messages indexes created")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error setting up indexes: {e}")
        return False

def print_mongodb_info():
    """Print MongoDB setup information"""
    print("\n" + "="*60)
    print("INGRES ChatBot - MongoDB Setup")
    print("="*60)
    print("This script verifies MongoDB connection and sets up the chat database.")
    print("\nMake sure you have:")
    print("1. MongoDB installed and running")
    print("2. Proper configuration in your .env file")
    print("3. Network access to the MongoDB instance")
    print("\nTo install MongoDB on macOS:")
    print("  brew tap mongodb/brew")
    print("  brew install mongodb-community")
    print("  brew services start mongodb-community")
    print("\nTo start MongoDB manually:")
    print("  mongod --config /opt/homebrew/etc/mongod.conf")
    print("="*60 + "\n")

def main():
    """Main function to run MongoDB setup and verification"""
    print_mongodb_info()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("Please copy .env.example to .env and configure your settings")
        return
    
    # Check MongoDB connection
    if not check_mongodb_connection():
        print("\n❌ MongoDB setup failed!")
        print("Please ensure MongoDB is running and accessible")
        return
    
    # Setup indexes
    print("\nSetting up database indexes...")
    if setup_mongodb_indexes():
        print("\n✅ MongoDB setup completed successfully!")
        print("Your INGRES ChatBot is ready to use MongoDB for chat storage.")
    else:
        print("\n⚠️  MongoDB connection works but index setup failed")
        print("The application should still work, but performance may be slower")

if __name__ == "__main__":
    main()