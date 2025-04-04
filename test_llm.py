#!/usr/bin/env python3
"""
Simple script to test only the GEMINI_API_KEY from your .env file.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Clear any existing GEMINI_API_KEY environment variable
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

# Load from .env file only
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"Loaded environment from: {env_path.absolute()}")
else:
    print(f"ERROR: .env file not found at: {env_path.absolute()}")
    sys.exit(1)

# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"GEMINI_API_KEY found: {api_key[:5]}...{api_key[-4:]}")
else:
    print("ERROR: No GEMINI_API_KEY found in .env file!")
    sys.exit(1)

# Test the API key
try:
    import google.generativeai as genai
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    print("Successfully configured Google Generative AI client")
    
    # Create the model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    # Test with a simple prompt
    print("Testing API with a simple prompt...")
    response = model.generate_content("Hello, give me a one sentence response as a test.")
    
    # Print response
    print("\nAPI Response:")
    print("-" * 40)
    print(response.text)
    print("-" * 40)
    print("\n✅ GEMINI_API_KEY is valid and working correctly!")
    
except Exception as e:
    print(f"\n❌ Error testing GEMINI_API_KEY: {e}")
    print("Please check your API key in the .env file.")