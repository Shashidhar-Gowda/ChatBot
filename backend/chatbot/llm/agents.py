# agents.py
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from typing import List, Dict, Any
import re
from difflib import get_close_matches
from .sql_agent import create_db_aware_agent
from langchain_core.messages import HumanMessage

# Import your existing tools
from .tools import (
    linear_regression_tool,
    polynomial_regression_tool,
    preprocess_tool,
    correlation_tool,
    # prediction_tool,
    get_report_generator_tool,
    get_ann_tool,
    # analyze_uploaded_file,
    get_history_tool,
    get_simple_summary_tool,
    get_dynamic_prediction_tool,
)

from .sql_agent import sql_tool

# ---------------- LLM Setup ----------------
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ---------------- Tool List ----------------
all_tools: List[Tool] = [
    linear_regression_tool,
    polynomial_regression_tool,
    preprocess_tool,
    correlation_tool,
    # prediction_tool,
    get_report_generator_tool,
    get_ann_tool,
    # analyze_uploaded_file,
    # sql_tool,
    get_history_tool,
    get_simple_summary_tool,
    get_dynamic_prediction_tool,
]

from langchain_core.prompts import PromptTemplate
summary_prompt = PromptTemplate.from_template("""
You are an assistant who summarizes tool outputs in plain English.
Summarize this result in a way that is helpful for a non-technical user.

Output:
{output}
""")

def wrap_with_summary(tool: Tool) -> Tool:
    def wrapped_fn(input_str: str) -> str:
        try:
            original_result = tool.invoke({"input_str": input_str})
            if isinstance(original_result, dict) and "output" in original_result:
                original_result = original_result["output"]
        except Exception as e:
            return f"Tool execution failed: {e}"

        prompt = summary_prompt.format(output=original_result)
        summary = llm.invoke(prompt).content
        return f"{original_result}\n\n\ud83d\udca1 Summary: {summary}"

    return Tool.from_function(name=tool.name, func=wrapped_fn, description=tool.description)

# Wrap tools with summaries
wrapped_tools = [
    wrap_with_summary(linear_regression_tool),
    wrap_with_summary(polynomial_regression_tool),
    wrap_with_summary(correlation_tool),
    # wrap_with_summary(prediction_tool),
    wrap_with_summary(get_ann_tool),
]

all_tools: List[Tool] = wrapped_tools
# ---------------- Prompt ----------------
default_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart agent helping with dataset analysis."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------------- LLM Fallback Agent ----------------
llm_agent_executor = AgentExecutor(
    agent=create_tool_calling_agent(llm=llm, tools=all_tools, prompt=default_prompt),
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True
)

# ---------------- Context-Aware Agent Router ----------------

def route_tool_by_intent(user_message: str, context: Dict[str, Any]) -> (AgentExecutor, str):
    message = user_message.lower()
    uploaded_files = context.get("uploaded_files", {})

    selected_tools = []
    matched_file_id = None
    matched_filename = None

    # Try matching uploaded file names
    if uploaded_files:
        # Exact match first
        for filename, file_id in uploaded_files.items():
            if filename.lower() in message:
                matched_file_id = file_id
                matched_filename = filename
                break

        # If no exact match, do fuzzy matching
        if not matched_file_id:
            filenames = list(uploaded_files.keys())
            close_matches = get_close_matches(message, filenames, n=1, cutoff=0.6)
            if close_matches:
                matched_filename = close_matches[0]
                matched_file_id = uploaded_files[matched_filename]

    # If multiple files uploaded but no match
    if not matched_file_id and len(uploaded_files) > 1:
        # Ask user to pick
        file_list = "\n".join([f"- {name}" for name in uploaded_files.keys()])
        system_message = f"You have uploaded multiple files:\n{file_list}\n\nPlease specify which file you'd like to analyze."

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return AgentExecutor(
            agent=create_tool_calling_agent(llm=llm, tools=[], prompt=fallback_prompt),
            tools=[],
            verbose=True,
            handle_parsing_errors=True
        ), user_message

    # Normal routing (if matched)
    if matched_file_id and matched_filename:
        context["file_id"] = str(matched_file_id)

        uploaded_file_path = uploaded_files.get(matched_filename)
        if uploaded_file_path:
            user_message = user_message.replace(matched_filename, uploaded_file_path)

    # Now intent-based tool selection
    if re.search(r'\b(linear regression|simple regression|regression equation)\b', message):
        selected_tools = [linear_regression_tool]
    elif re.search(r'\b(polynomial regression|curve fitting|nonlinear regression)\b', message):
        selected_tools = [polynomial_regression_tool]
    elif re.search(r'\b(preprocess|clean|missing values|data cleaning)\b', message):
        selected_tools = [preprocess_tool]
    elif re.search(r'\b(correlation|pearson|spearman|relationship)\b', message):
        selected_tools = [correlation_tool]
    elif re.search(r'\b(predict|forecast|future estimate|prediction)\b', message):
        selected_tools = [get_dynamic_prediction_tool]
    elif re.search(r'\b(insight|reporting|report)\b', message):
        selected_tools = [get_report_generator_tool]
    elif re.search(r'\b(average|total|higest|minimum|enteries)\b', message):
        selected_tools = [get_history_tool]
    elif re.search(r'\b(classification|classify|ann|neural network)\b', message):
        selected_tools = [get_ann_tool]
    elif re.search(r'\b(summary|summarize)\b', message):
        selected_tools = [get_simple_summary_tool]    
    # elif re.search(r'\b(sql|database|query)\b', message):
    #     return create_db_aware_agent(llm), user_message

    if selected_tools:
        return AgentExecutor(
            agent=create_tool_calling_agent(llm=llm, tools=selected_tools, prompt=default_prompt),
            tools=selected_tools,
            verbose=True,
            handle_parsing_errors=True
        ), user_message

    # Default fallback
    return llm_agent_executor, user_message
