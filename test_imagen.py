#!/usr/bin/env python3
"""
Test script for Google Gemini Imagen 4 image generation
"""

from google import genai
from google.genai import types
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imagen_generation():
    """Test Imagen 4 image generation"""
    print("🧪 Testing Imagen 4 image generation...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not found or invalid")
        print("   Please set GEMINI_API_KEY in your .env file")
        return False
    
    try:
        # Initialize client
        c = genai.Client()
        
        # Get model from environment
        model_id = os.getenv("MODEL_IMAGE", "imagen-4.0-generate-001")
        print(f"📸 Using model: {model_id}")
        
        # Generate test image
        print("🎨 Generating test image...")
        r = c.models.generate_images(
            model=model_id,
            prompt="storybook illustration of a friendly talking compass, soft pastel",
            config=types.GenerateImagesConfig(
                number_of_images=1, 
                imageSize="1K", 
                aspect_ratio="1:1"
            ),
        )
        
        # Save test image
        path = pathlib.Path("imagen_smoke_test.png")
        path.write_bytes(r.generated_images[0].image.image_bytes)
        print(f"✅ Saved test image: {path}")
        print(f"   File size: {path.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Imagen test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imagen_generation()
    exit(0 if success else 1)
