"""
Test script for LangChain integration with Hermes Function Calling

This script demonstrates how to use Hermes tools with LangChain agents.
"""

import sys
import os

# Add the parent directory to the Python path to import Hermes modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Conditional imports to handle environments where LangChain is not available
try:
    from langchain.llms import Ollama
    from langchain_integration import HermesToolAdapter, run_hermes_with_langchain

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain is not available. Please install it to run this example.")
    print("You can install it by running: pip install langchain")


def test_langchain_integration():
    """Test the LangChain integration with a simple query"""
    if not LANGCHAIN_AVAILABLE:
        print("Skipping test as LangChain is not available.")
        return

    # Initialize LLM (using Ollama as an example)
    # Make sure you have Ollama running and the model pulled
    try:
        llm = Ollama(model="phi3")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        print("Make sure Ollama is running and the model is available.")
        return

    # Test query
    query = "What's the current stock price of Apple (AAPL)?"

    try:
        # Run the query using Hermes tools through LangChain
        result = run_hermes_with_langchain(llm, query)
        print(f"Query: {query}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error running query: {e}")


if __name__ == "__main__":
    test_langchain_integration()
