#!/usr/bin/env python3
"""
Test script to verify the new project structure works correctly.
"""

import sys
import os
import subprocess
from typing import List

def test_imports() -> bool:
    """Test that all modules can be imported correctly."""
    print("🔍 Testing module imports...")
    
    try:
        # Test UI imports
        from ui.streamlit_app import main as ui_main
        print("  ✅ UI module imports successfully")
        
        # Test export imports
        from export.cli_interface import CommandLineInterface
        from export.csv_exporter import CSVExporter
        from export.export_patterns import export_patterns_to_csv
        print("  ✅ Export modules import successfully")
        
        # Test main entry point
        import main
        print("  ✅ Main entry point imports successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_cli_help() -> bool:
    """Test that CLI help works."""
    print("🔍 Testing CLI help...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and 'Stock Pattern Detector' in result.stdout:
            print("  ✅ Main CLI help works")
            return True
        else:
            print(f"  ❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ CLI help error: {e}")
        return False


def test_export_help() -> bool:
    """Test that export CLI help works."""
    print("🔍 Testing export CLI help...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'export/export_cli.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and 'CSV Export Tool' in result.stdout:
            print("  ✅ Export CLI help works")
            return True
        else:
            print(f"  ❌ Export CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Export CLI help error: {e}")
        return False


def test_directory_structure() -> bool:
    """Test that required directories exist."""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'ui',
        'export', 
        'data',
        'patterns',
        'utils',
        'visualization'
    ]
    
    required_files = [
        'main.py',
        'ui/streamlit_app.py',
        'export/cli_interface.py',
        'export/csv_exporter.py',
        'export/export_cli.py',
        'export/export_patterns.py'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_items.append(f"Directory: {dir_name}")
    
    # Check files
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_items.append(f"File: {file_name}")
    
    if missing_items:
        print(f"  ❌ Missing items: {', '.join(missing_items)}")
        return False
    else:
        print("  ✅ All required directories and files exist")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Testing New Project Structure")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("CLI Help", test_cli_help),
        ("Export CLI Help", test_export_help)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The new structure is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)