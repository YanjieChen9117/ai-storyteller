#!/usr/bin/env python3
"""
Test script to verify your AI Storyteller setup is working correctly.
Run this before starting the main app to check your configuration.

Usage: python test_setup.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_environment():
    """Test if environment variables are set correctly."""
    print("🔍 Testing environment setup...")
    
    # Load .env file
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with your API key")
        return False
    elif api_key == "sk-your-api-key-here":
        print("❌ Please replace the placeholder API key with your actual key")
        return False
    else:
        print("✅ OPENAI_API_KEY found")
    
    # Check model settings
    model_text = os.getenv("MODEL_TEXT", "gpt-4o-mini")
    model_image = os.getenv("MODEL_IMAGE", "dall-e-3")
    image_size = os.getenv("IMAGE_SIZE", "1024x1024")
    
    print(f"✅ MODEL_TEXT: {model_text}")
    print(f"✅ MODEL_IMAGE: {model_image}")
    print(f"✅ IMAGE_SIZE: {image_size}")
    
    return True

def test_dependencies():
    """Test if all required packages are installed."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        "streamlit",
        "openai", 
        "pydantic",
        "python-dotenv",
        "Pillow",
        "fpdf2",
        "tenacity"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def test_api_connection():
    """Test if we can connect to OpenAI API."""
    print("\n🌐 Testing API connection...")
    
    try:
        from openai import OpenAI
        from utils import llm_text
        
        # Simple test call
        response = llm_text("Say 'Hello, AI Storyteller!'", temperature=0.1)
        
        if "hello" in response.lower() or "ai storyteller" in response.lower():
            print("✅ API connection successful")
            print(f"   Response: {response}")
            return True
        else:
            print("⚠️  API responded but with unexpected content")
            print(f"   Response: {response}")
            return True
            
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        print("   Check your API key and internet connection")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Storyteller Setup Test")
    print("=" * 40)
    
    # Test environment
    env_ok = test_environment()
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Only test API if environment is good
    api_ok = False
    if env_ok and deps_ok:
        api_ok = test_api_connection()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   API Connection: {'✅' if api_ok else '❌'}")
    
    if env_ok and deps_ok and api_ok:
        print("\n🎉 All tests passed! You're ready to run:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
        sys.exit(1)

if __name__ == "__main__":
    main()
