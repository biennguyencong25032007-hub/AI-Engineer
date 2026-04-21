"""Tests for agent modules"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.react_agent import ReActAgent
from agents.tool_calling_agent import ToolCallingAgent


def test_base_agent():
    """Test BaseAgent abstract class"""
    print("Testing BaseAgent...")
    # BaseAgent is abstract, should not be instantiated
    try:
        agent = BaseAgent(lambda x: x)
        print("✗ BaseAgent should not be instantiable")
        return False
    except TypeError:
        print("✓ BaseAgent correctly abstract")
        return True


def test_react_agent():
    """Test ReActAgent"""
    print("\nTesting ReActAgent...")
    agent = ReActAgent(lambda prompt: f"Response to: {prompt[:50]}...")
    
    result = agent.run("What is 2+2?")
    assert isinstance(result, str), "Result should be string"
    assert len(result) > 0, "Result should not be empty"
    
    print("✓ ReActAgent works correctly")
    return True


def test_tool_calling_agent():
    """Test ToolCallingAgent"""
    print("\nTesting ToolCallingAgent...")
    
    def mock_llm(prompt):
        return '{"tool": "calculator", "input": "2+2"}'
    
    tools = {
        'calculator': lambda x: "4"
    }
    
    agent = ToolCallingAgent(mock_llm, tools)
    result = agent.run("Calculate 2+2")
    
    assert result == "4", f"Expected '4', got '{result}'"
    print("✓ ToolCallingAgent works correctly")
    return True


def run_all_tests():
    """Run all agent tests"""
    print("=" * 50)
    print("RUNNING AGENT TESTS")
    print("=" * 50)
    
    tests = [
        test_base_agent,
        test_react_agent,
        test_tool_calling_agent
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
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
