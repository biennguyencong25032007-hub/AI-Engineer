"""Tests for tool modules"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.calculator_tool import CalculatorTool
from tools.file_tool import FileTool
from tools.search_tool import SearchTool
from tools.sql_tool import SQLTool
import tempfile
import os


def test_calculator_tool():
    """Test CalculatorTool"""
    print("\nTesting CalculatorTool...")
    calc = CalculatorTool()
    
    # Test basic arithmetic
    result = calc("2 + 2")
    assert "4" in result, f"Expected '4' in result, got '{result}'"
    
    result = calc("10 * 5")
    assert "50" in result, f"Expected '50' in result, got '{result}'"
    
    result = calc("sqrt(16)")
    assert "4" in result, f"Expected '4' in result, got '{result}'"
    
    # Test error handling
    result = calc("invalid expression")
    assert "Error" in result, f"Expected error for invalid expression"
    
    print("✓ CalculatorTool works correctly")
    return True


def test_file_tool():
    """Test FileTool"""
    print("\nTesting FileTool...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_tool = FileTool(base_dir=tmpdir)
        
        # Test write
        test_content = "Hello, World!"
        result = file_tool("write", path="test.txt", content=test_content)
        assert "Successfully" in result, f"Write failed: {result}"
        
        # Test read
        content = file_tool("read", path="test.txt")
        assert content == test_content, f"Read returned unexpected content: {content}"
        
        # Test exists
        exists = file_tool("exists", path="test.txt")
        assert exists == "True", f"Exists check failed: {exists}"
        
        # Test list
        listing = file_tool("list", directory=".")
        assert "test.txt" in listing, f"File not in listing: {listing}"
        
        # Test delete
        result = file_tool("delete", path="test.txt")
        assert "Successfully" in result, f"Delete failed: {result}"
        
        # Verify deleted
        exists = file_tool("exists", path="test.txt")
        assert exists == "False", f"File still exists after delete"
    
    print("✓ FileTool works correctly")
    return True


def test_search_tool():
    """Test SearchTool"""
    print("\nTesting SearchTool...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test documents
        docs_dir = Path(tmpdir) / "documents"
        docs_dir.mkdir()
        
        test_file = docs_dir / "test.txt"
        test_file.write_text("Python is a programming language.\nAI is interesting.")
        
        search_tool = SearchTool(data_dir=tmpdir)
        
        # Test search
        result = search_tool("Python")
        assert "Python" in result or "Found" in result, f"Search failed: {result}"
        
        result = search_tool("nonexistent term xyz")
        assert "No results" in result or "not found" in result.lower(), f"Expected no results: {result}"
    
    print("✓ SearchTool works correctly")
    return True


def test_sql_tool():
    """Test SQLTool"""
    print("\nTesting SQLTool...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        sql_tool = SQLTool(db_path=db_path)
        
        # Test create table
        result = sql_tool("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        assert "created" in result.lower() or "Success" in result, f"Create table failed: {result}"
        
        # Test insert
        result = sql_tool("INSERT INTO test (name) VALUES ('Alice')")
        assert "Inserted" in result or "Success" in result, f"Insert failed: {result}"
        
        # Test select
        result = sql_tool("SELECT * FROM test")
        assert "Alice" in result or "Columns" in result, f"Select failed: {result}"
        
        # Test schema
        schema = sql_tool.get_schema()
        assert "test" in schema, f"Schema doesn't contain table: {schema}"
        
        print("✓ SQLTool works correctly")
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests():
    """Run all tool tests"""
    print("=" * 50)
    print("RUNNING TOOL TESTS")
    print("=" * 50)
    
    tests = [
        test_calculator_tool,
        test_file_tool,
        test_search_tool,
        test_sql_tool
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
