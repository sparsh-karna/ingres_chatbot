"""
Command Line Interface for INGRES ChatBot
Test and interact with the system via command line
"""

import argparse
import sys
import os
from database_manager import DatabaseManager, DataLoader
from query_processor import QueryProcessor
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_database():
    """Setup and populate the database with CSV data"""
    print("🔄 Setting up database and loading CSV data...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize data loader
        data_loader = DataLoader(db_manager)
        
        # Load all CSV files
        data_loader.load_all_csv_files()
        
        # Get and display database summary
        summary = data_loader.get_database_summary()
        print("\n✅ Database setup completed successfully!")
        print("\n=== Database Summary ===")
        
        total_rows = 0
        for table_name, info in summary.items():
            year = table_name.replace('groundwater_data_', '').replace('_', '-')
            print(f"📊 Year {year}: {info['row_count']:,} records, {len(info['schema']['columns'])} columns")
            total_rows += info['row_count']
        
        print(f"\n📈 Total records in database: {total_rows:,}")
        
        # Close connection
        db_manager.close_connection()
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False


def interactive_chat():
    """Start interactive chat session"""
    print("🤖 Starting INGRES AI ChatBot...")
    print("Type 'quit', 'exit', or 'bye' to end the session")
    print("Type 'help' for sample questions")
    print("=" * 60)
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        
        print("✅ ChatBot initialized successfully!\n")
        
        # Sample questions for help
        sample_questions = [
            "What are the top 5 states with highest groundwater recharge in 2024-2025?",
            "Show me groundwater extraction data for Andhra Pradesh",
            "Which districts have the highest rainfall in 2023-2024?",
            "Compare groundwater levels between 2020 and 2024",
            "What is the average groundwater recharge in Maharashtra?",
        ]
        
        while True:
            try:
                # Get user input
                user_question = input("\n💬 Your question: ").strip()
                
                # Check for exit commands
                if user_question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thank you for using INGRES ChatBot.")
                    break
                
                # Show help
                if user_question.lower() == 'help':
                    print("\n💡 Sample questions you can ask:")
                    for i, question in enumerate(sample_questions, 1):
                        print(f"{i}. {question}")
                    continue
                
                # Skip empty input
                if not user_question:
                    print("⚠️ Please enter a question.")
                    continue
                
                print(f"🔍 Processing: {user_question}")
                print("-" * 60)
                
                # Process the query
                result = query_processor.process_user_query(user_question)
                
                if result['success']:
                    print(f"\n🔧 Generated SQL Query:")
                    print(f"```sql\n{result['sql_query']}\n```")
                    
                    print(f"\n📊 Results:")
                    if not result['data'].empty:
                        print(f"Found {len(result['data'])} records")
                        
                        # Show first few rows if data exists
                        if len(result['data']) <= 5:
                            print("\nComplete results:")
                            print(result['data'].to_string(index=False))
                        else:
                            print("\nSample results (first 5 rows):")
                            print(result['data'].head().to_string(index=False))
                            print(f"... and {len(result['data']) - 5} more rows")
                    
                    print(f"\n🤖 AI Response:")
                    print(result['response'])
                
                else:
                    print(f"❌ Error: {result['error']}")
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        print(f"❌ Failed to initialize ChatBot: {e}")
        sys.exit(1)


def test_single_query(question: str):
    """Test a single query"""
    print(f"🔍 Testing query: {question}")
    print("-" * 60)
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        query_processor = QueryProcessor(db_manager)
        
        # Process the query
        result = query_processor.process_user_query(question)
        
        if result['success']:
            print(f"\n✅ Query processed successfully!")
            print(f"\n🔧 Generated SQL Query:")
            print(result['sql_query'])
            
            print(f"\n📊 Results: {len(result['data'])} records")
            if not result['data'].empty and len(result['data']) <= 10:
                print(result['data'].to_string(index=False))
            elif not result['data'].empty:
                print(result['data'].head().to_string(index=False))
                print(f"... and {len(result['data']) - 5} more rows")
            
            print(f"\n🤖 AI Response:")
            print(result['response'])
        
        else:
            print(f"❌ Error: {result['error']}")
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")


def check_database_status():
    """Check database connection and table status"""
    print("🔍 Checking database status...")
    
    try:
        db_manager = DatabaseManager()
        
        # Check connection
        with db_manager.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection: OK")
        
        # Check tables
        with db_manager.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'groundwater_data_%'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            
            if tables:
                print(f"✅ Found {len(tables)} data tables:")
                for table in tables:
                    year = table.replace('groundwater_data_', '').replace('_', '-')
                    
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    row_count = count_result.fetchone()[0]
                    
                    print(f"  📊 {year}: {row_count:,} records")
            else:
                print("⚠️ No data tables found. Run 'python cli.py --setup' first.")
        
        db_manager.close_connection()
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="INGRES AI ChatBot - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --setup                    # Setup database and load CSV data
  python cli.py --chat                     # Start interactive chat session
  python cli.py --status                   # Check database status
  python cli.py --query "Your question"    # Test a single query
        """
    )
    
    parser.add_argument('--setup', action='store_true', 
                       help='Setup database and load CSV data')
    parser.add_argument('--chat', action='store_true', 
                       help='Start interactive chat session')
    parser.add_argument('--status', action='store_true', 
                       help='Check database connection and status')
    parser.add_argument('--query', type=str, 
                       help='Test a single query')
    
    args = parser.parse_args()
    
    # Check if environment file exists
    if not os.path.exists('.env'):
        print("⚠️ No .env file found. Please create one based on .env.example")
        print("Make sure to configure your database and Google API key settings.")
        return
    
    # Execute based on arguments
    if args.setup:
        setup_database()
    elif args.chat:
        interactive_chat()
    elif args.status:
        check_database_status()
    elif args.query:
        test_single_query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()