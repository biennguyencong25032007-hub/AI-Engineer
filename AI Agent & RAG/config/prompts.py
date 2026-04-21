"""Prompt templates for agents and RAG system"""

# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant. Use the following context to answer questions accurately.

If you don't know the answer, say so honestly. Always cite your sources when available."""

REACT_SYSTEM_PROMPT = """You are a ReAct (Reasoning + Acting) agent. You have access to tools to help answer questions.

Follow this format:
Thought: I need to...
Action: Use tool X with input Y
Observation: Result from tool
Thought: Now I understand...
Final Answer: Your final response

Be systematic and thorough."""

TOOL_CALLING_SYSTEM_PROMPT = """You are an AI assistant with access to tools. When you need information or need to perform an action, use the appropriate tool.

Always think step by step before calling tools. Provide clear and concise answers."""

# RAG prompts
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

RAG_REFINE_PROMPT_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be standalone.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

# Agent prompts
AGENT_EXECUTION_PROMPT = """You are an AI agent. Your task is to help the user with their query.

User Query: {query}
Available Tools:
{tools}

Think carefully about which tool to use and how to use it."""

# Evaluation prompts
EVALUATION_PROMPT = """Evaluate the following answer against the ground truth.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Score from 1-10 and provide brief reasoning."""

# Memory prompts
MEMORY_SUMMARY_PROMPT = """Summarize the following interaction:

User: {user_input}
Assistant: {assistant_output}

Key points:"""

# Tool-specific prompts
CALCULATOR_PROMPT = """Perform the calculation: {expression}

Show step-by-step reasoning."""

SEARCH_PROMPT = """Search query: {query}

Find relevant and recent information."""

FILE_OPERATION_PROMPT = """Operation: {operation}
File path: {file_path}
Content (if any): {content}

Perform the operation safely and confirm completion."""

SQL_PROMPT = """SQL Task: {task}
Database schema: {schema}

Write and execute appropriate SQL query."""
