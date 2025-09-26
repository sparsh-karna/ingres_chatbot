"""
Setup script for Twilio Voice Integration
This script helps configure Twilio for voice calls
"""

import os
from dotenv import load_dotenv

load_dotenv()

def print_setup_instructions():
    """Print setup instructions for Twilio voice integration"""
    
    print("=" * 60)
    print("TWILIO VOICE INTEGRATION SETUP")
    print("=" * 60)
    
    print("\n1. START THE VOICE SERVICE:")
    print("   python twilio_voice.py")
    print("   (Runs on port 8002)")
    
    print("\n2. EXPOSE WITH NGROK:")
    print("   ngrok http 8002")
    print("   (Copy the HTTPS URL, e.g., https://abc123.ngrok-free.app)")
    
    print("\n3. CONFIGURE TWILIO CONSOLE:")
    print("   Go to: https://console.twilio.com/")
    print("   Navigate to: Phone Numbers → Manage → Active numbers")
    print("   Click on your phone number: +12315295447")
    
    print("\n4. SET VOICE WEBHOOK:")
    print("   In 'Voice' section:")
    print("   - A call comes in: POST https://YOUR_NGROK_URL/twilio/voice/incoming")
    print("   - Method: POST")
    print("   - Save configuration")
    
    print("\n5. TEST THE INTEGRATION:")
    print("   - Call +12315295447 from any phone")
    print("   - You'll hear: 'Hello! Welcome to INGRES Groundwater Analysis System...'")
    print("   - Speak your question after the beep")
    print("   - Wait for the AI response")
    
    print("\n6. REQUIRED ENVIRONMENT VARIABLES:")
    required_vars = [
        "SARVAM_API_KEY",
        "GOOGLE_TRANSLATION_API", 
        "AZURE_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    print("   Make sure these are set in your .env file:")
    for var in required_vars:
        value = os.getenv(var)
        status = "✓ SET" if value else "✗ MISSING"
        print(f"   - {var}: {status}")
    
    print("\n7. VOICE PIPELINE FLOW:")
    print("   Call → Greeting → Record Speech → STT → Query Processing → TTS → Response")
    
    print("\n8. SUPPORTED LANGUAGES:")
    print("   - Hindi (hi-IN)")
    print("   - English (en-IN)")
    print("   - Tamil (ta-IN)")
    print("   - Telugu (te-IN)")
    print("   - Bengali (bn-IN)")
    print("   - Gujarati (gu-IN)")
    print("   - Kannada (kn-IN)")
    print("   - Malayalam (ml-IN)")
    print("   - Marathi (mr-IN)")
    print("   - Punjabi (pa-IN)")
    
    print("\n9. TROUBLESHOOTING:")
    print("   - Check logs in terminal for errors")
    print("   - Verify all API keys are correct")
    print("   - Ensure ngrok is running and URL is updated in Twilio")
    print("   - Test with simple questions first")
    
    print("\n10. EXAMPLE QUERIES TO TRY:")
    print("    - 'What is the groundwater recharge in Gujarat?'")
    print("    - 'Show me rainfall data for Tamil Nadu'")
    print("    - 'Compare extraction rates between states'")
    print("    - 'How many years of data do you have?'")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE! Your voice chatbot is ready.")
    print("=" * 60)

if __name__ == "__main__":
    print_setup_instructions()
