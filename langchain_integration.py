"""
LangChain integration for Hermes Function Calling

This module provides integration between Hermes Function Calling and LangChain,
allowing Hermes tools to be used within LangChain agents and vice versa.
"""

import json
from typing import List, Dict, Any

# Conditional imports to handle environments where LangChain is not available
try:
    from langchain.tools import BaseTool
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseTool = object
    LANGCHAIN_AVAILABLE = False

# Import Hermes functions
import functions
from functions import get_openai_tools


class HermesToolAdapter:
    """Adapter to convert Hermes tools to LangChain tools"""

    def __init__(self):
        # Check if LangChain is available
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not available. Please install it to use this feature."
            )

        # Get all Hermes tools
        self.tools = functions.get_langchain_tools()

    def get_tools(self) -> List[BaseTool]:
        """Get all Hermes tools as LangChain tools"""
        return self.tools


def create_hermes_agent(llm, tools: List[BaseTool]):
    """
    Create a LangChain agent that can use Hermes tools

    Args:
        llm: Language model to use for the agent
        tools: List of LangChain tools (including Hermes tools)

    Returns:
        AgentExecutor: Configured agent executor
    """
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not available. Please install it to use this feature."
        )

    # Create prompt for tool calling agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with access to various tools."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a Hermes tool call

    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool

    Returns:
        Dict containing the tool result
    """
    # Get all Hermes tools
    tools = functions.get_langchain_tools()
    tool_dict = {tool.name: tool for tool in tools}

    # Find the tool
    if tool_name not in tool_dict:
        return {
            "name": tool_name,
            "error": f"Tool '{tool_name}' not found",
            "content": None,
        }

    # Execute the tool
    try:
        tool = tool_dict[tool_name]
        result = tool.invoke(arguments)
        return {"name": tool_name, "content": result, "error": None}
    except Exception as e:
        return {"name": tool_name, "error": str(e), "content": None}


def convert_hermes_result_to_langchain(tool_name: str, result: Dict[str, Any]) -> Any:
    """
    Convert Hermes tool result to LangChain ToolMessage

    Args:
        tool_name: Name of the tool that was executed
        result: Result from the tool execution

    Returns:
        ToolMessage: LangChain ToolMessage
    """
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not available. Please install it to use this feature."
        )

    if result.get("error"):
        content = f"Error executing {tool_name}: {result['error']}"
    else:
        content = str(result.get("content", ""))

    return ToolMessage(
        content=content,
        tool_call_id=tool_name,  # Simplified - in practice, you'd use a proper ID
    )


# Example usage function
def run_hermes_with_langchain(llm, query: str):
    """
    Run a query using Hermes tools through LangChain agent

    Args:
        llm: Language model to use
        query: User query to process

    Returns:
        Dict containing the final result
    """
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not available. Please install it to use this feature."
        )

    # Get Hermes tools
    adapter = HermesToolAdapter()
    tools = adapter.get_tools()

    # Create agent
    agent_executor = create_hermes_agent(llm, tools)

    # Execute query
    result = agent_executor.invoke({"input": query})

    return result


# Fallback functions when LangChain is not available
def get_langchain_tools_fallback():
    """Fallback function to get tools when LangChain is not available"""
    return []


def create_hermes_agent_fallback(llm, tools):
    """Fallback function to create agent when LangChain is not available"""
    raise ImportError(
        "LangChain is not available. Please install it to use this feature."
    )


def execute_tool_call_fallback(
    tool_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Fallback function to execute tool call when LangChain is not available"""
    raise ImportError(
        "LangChain is not available. Please install it to use this feature."
    )


def convert_hermes_result_to_langchain_fallback(
    tool_name: str, result: Dict[str, Any]
) -> Any:
    """Fallback function to convert result when LangChain is not available"""
    raise ImportError(
        "LangChain is not available. Please install it to use this feature."
    )


def run_hermes_with_langchain_fallback(llm, query: str):
    """Fallback function to run Hermes with LangChain when LangChain is not available"""
    raise ImportError(
        "LangChain is not available. Please install it to use this feature."
    )
