"""
Vercel deployment entry point for INGRES ChatBot FastAPI application
"""

from app_modular import app
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = app
