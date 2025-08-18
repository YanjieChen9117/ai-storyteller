#!/usr/bin/env python3
"""
Test script for export functionality
"""
import json
from pathlib import Path
from app import export_pdf, export_zip

def test_export_functions():
    """Test PDF and ZIP export functions."""
    print("🧪 Testing export functions...")
    
    # Create test data
    test_pages = [
        {
            "page": 1,
            "summary": "Test page 1",
            "text": "This is a test page for export functionality testing.",
            "image_prompt_final": "Test image prompt 1"
        },
        {
            "page": 2,
            "summary": "Test page 2", 
            "text": "Another test page to verify export works correctly.",
            "image_prompt_final": "Test image prompt 2"
        }
    ]
    
    # Create test directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Create test images directory
    images_dir = test_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create dummy image files
    for i in range(1, 3):
        img_path = images_dir / f"page_{i:02d}.png"
        # Create a simple text file as placeholder
        img_path.write_text(f"Test image {i}")
    
    try:
        # Test PDF export
        print("📄 Testing PDF export...")
        pdf_path = export_pdf(test_dir, test_pages, "512x512")
        print(f"✅ PDF created: {pdf_path}")
        print(f"   File exists: {pdf_path.exists()}")
        print(f"   File size: {pdf_path.stat().st_size} bytes")
        
        # Test ZIP export
        print("\n📁 Testing ZIP export...")
        zip_path = export_zip(test_dir)
        print(f"✅ ZIP created: {zip_path}")
        print(f"   File exists: {zip_path.exists()}")
        print(f"   File size: {zip_path.stat().st_size} bytes")
        
        print("\n🎉 All export tests passed!")
        
    except Exception as e:
        print(f"❌ Export test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test files
        print("\n🧹 Cleaning up test files...")
        try:
            if 'pdf_path' in locals() and pdf_path.exists():
                pdf_path.unlink()
                print("   PDF file removed")
            if 'zip_path' in locals() and zip_path.exists():
                zip_path.unlink()
                print("   ZIP file removed")
            
            # Remove test images
            for img_file in images_dir.glob("*.png"):
                img_file.unlink()
            images_dir.rmdir()
            test_dir.rmdir()
            print("   Test directories removed")
            
        except Exception as cleanup_error:
            print(f"   Warning: Cleanup failed: {cleanup_error}")

if __name__ == "__main__":
    test_export_functions()
