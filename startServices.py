#!/usr/bin/env python3
"""
Start All Agentic System Services
Launches all 8 microservices for the Enhanced Agentic Policy Analysis System
Includes WhatsApp chatbot for Twilio integration
"""

import subprocess
import time
import sys
import os
import signal
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# Service configurations
SERVICES = [
    {
        "name": "Query Processor",
        "file": "queryProcessor.py",
        "port": 8001,
        "description": "Generates Python code from natural language queries"
    },
    {
        "name": "Code Executor", 
        "file": "codeExecutor.py",
        "port": 8002,
        "description": "Executes Python code safely for CSV analysis"
    },
    {
        "name": "Result Analyzer",
        "file": "resultAnalyzer.py", 
        "port": 8003,
        "description": "Converts results into natural language explanations"
    },
    {
        "name": "Orchestrator Agent",
        "file": "orchestratorAgent.py",
        "port": 8004,
        "description": "Breaks complex queries into subtasks (CSV-AWARE)"
    },
    {
        "name": "Task Execution Engine",
        "file": "taskExecutionEngine.py",
        "port": 8005,
        "description": "Manages and executes tasks automatically"
    },
    {
        "name": "Report Generator",
        "file": "reportGenerator.py",
        "port": 8006,
        "description": "Generates comprehensive policy reports"
    },
    {
        "name": "Web Research Agent",
        "file": "webResearchAgent.py",
        "port": 8007,
        "description": "Conducts simulated web research"
    }
]

# Global variables for process management
processes = []
shutdown_event = threading.Event()

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("Jal-Drishti")
    print("=" * 80)
    print("Starting 8 AI Microservices")
    print("CSV-Aware | Modern UI | Government-Ready Reports | WhatsApp Bot")
    print("=" * 80)

def check_port(port):
    """Check if a port is already in use"""
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return True
    except:
        return False

def kill_existing_processes():
    """Kill any existing processes on our ports"""
    print("Checking for existing services...")

    for service in SERVICES:
        if check_port(service["port"]):
            print(f"   Port {service['port']} in use, attempting to free...")
            try:
                # Try to kill processes using the port
                subprocess.run(f"lsof -ti:{service['port']} | xargs kill -9",
                             shell=True, capture_output=True)
                time.sleep(1)
            except:
                pass

def start_service(service):
    """Start a single service"""
    try:
        print(f"Starting {service['name']} on port {service['port']}...")

        # Check if file exists
        if not os.path.exists(service['file']):
            print(f"   [ERROR] File {service['file']} not found!")
            return None

        # Start the process
        process = subprocess.Popen(
            [sys.executable, service['file']],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a moment for startup
        time.sleep(2)

        # Check if process is still running
        if process.poll() is None:
            print(f"   [OK] {service['name']} started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"   [ERROR] {service['name']} failed to start")
            if stderr:
                print(f"      Error: {stderr[:200]}...")
            return None

    except Exception as e:
        print(f"   [ERROR] Error starting {service['name']}: {str(e)}")
        return None

def health_check():
    """Check health of all services"""
    print("\nPerforming health checks...")
    healthy_services = 0
    
    for service in SERVICES:
        try:
            response = requests.get(f"http://localhost:{service['port']}/", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {service['name']} - Healthy")
                healthy_services += 1
            else:
                print(f"   ⚠️  {service['name']} - Responding but status {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"   ❌ {service['name']} - Not responding")
    
    return healthy_services

def monitor_services():
    """Monitor services and restart if needed"""
    print("\nStarting service monitor...")
    
    while not shutdown_event.is_set():
        time.sleep(30)  # Check every 30 seconds
        
        if shutdown_event.is_set():
            break
            
        failed_services = []
        for i, service in enumerate(SERVICES):
            if i < len(processes) and processes[i] and processes[i].poll() is not None:
                failed_services.append((i, service))
        
        if failed_services:
            print(f"\n⚠️  Detected {len(failed_services)} failed services, restarting...")
            for i, service in failed_services:
                print(f"   🔄 Restarting {service['name']}...")
                processes[i] = start_service(service)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n\nShutdown signal received...")
    shutdown_event.set()
    cleanup_and_exit()

def cleanup_and_exit():
    """Clean shutdown of all services"""
    print("🧹 Cleaning up services...")
    
    for i, process in enumerate(processes):
        if process and process.poll() is None:
            service_name = SERVICES[i]['name'] if i < len(SERVICES) else f"Service {i}"
            print(f"   🛑 Stopping {service_name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    print("✅ All services stopped")
    print("\nAgentic System Shutdown Complete!")
    sys.exit(0)

def print_service_info():
    """Print service information"""
    print("\nSERVICE OVERVIEW:")
    print("-" * 80)
    for service in SERVICES:
        print(f"🔹 {service['name']:<25} | Port {service['port']} | {service['description']}")
    print("-" * 80)

def print_access_info():
    """Print access information"""
    print("\nACCESS INFORMATION:")
    print("-" * 80)
    print("Agentic Interface: file:///Users/akshatmajila/SIH/agentic_interface.html")
    print("Original Interface:  file:///Users/akshatmajila/SIH/query_interface.html")
    print("Service Health:      http://localhost:8001/ (or any service port)")
    print("WhatsApp Bot:        http://localhost:5001/test (Test endpoint)")
    print("WhatsApp Webhook:    http://localhost:5001/webhook (For Twilio)")
    print("-" * 80)
    
    print("\nDEMO QUERIES:")
    print("🔹 'Analyze groundwater availability in northeastern Indian states'")
    print("🔹 'Prepare a Plan to increase groundwater levels in northeastern India'")
    print("🔹 'Compare rainfall and groundwater recharge across Indian regions'")

def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print_banner()
        print_service_info()
        
        # Kill existing processes
        kill_existing_processes()
        
        # Start all services
        print("\nSTARTING SERVICES:")
        print("-" * 80)
        
        for service in SERVICES:
            process = start_service(service)
            processes.append(process)
            time.sleep(1)  # Stagger startup
        
        # Health check
        time.sleep(5)  # Wait for all services to fully start
        healthy_count = health_check()
        
        if healthy_count == len(SERVICES):
            print(f"\nALL {len(SERVICES)} SERVICES STARTED SUCCESSFULLY!")
            print_access_info()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=monitor_services, daemon=True)
            monitor_thread.start()
            
            print("\n⌨️  Press Ctrl+C to stop all services")
            print("Services are being monitored and will auto-restart if needed")
            
            # Keep main thread alive
            while not shutdown_event.is_set():
                time.sleep(1)
                
        else:
            print(f"\n⚠️  Only {healthy_count}/{len(SERVICES)} services started successfully")
            print("❌ Some services failed to start. Check the errors above.")
            
            # Still show access info for working services
            if healthy_count > 0:
                print_access_info()
                print("\n⌨️  Press Ctrl+C to stop running services")
                
                # Keep running with partial services
                while not shutdown_event.is_set():
                    time.sleep(1)
            else:
                cleanup_and_exit()
                
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received...")
        cleanup_and_exit()
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        cleanup_and_exit()

if __name__ == "__main__":
    main()
