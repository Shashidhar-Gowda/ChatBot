# agents.py
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from typing import List, Dict, Any
import re
from difflib import get_close_matches
from .sql_agent import create_db_aware_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    get_visualization_tool,
)

from .sql_agent import sql_tool

# ---------------- LLM Setup ----------------
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

def clean_surrogates(text: str) -> str:
    """Removes surrogate characters from a string."""
    return text.encode('utf-8', 'ignore').decode('utf-8')

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
    get_visualization_tool
]

from langchain_core.prompts import PromptTemplate


summary_prompt = PromptTemplate.from_template("""
You are an expert data analyst.
The user asked a question, and a tool provided the following result.
Your job is to provide a concise and complete answer to the user's question,
using the tool's result. Do not mention the tool itself.

User's Question: {user_question}
Tool's Result: {tool_result}

Final Answer:
""")

def wrap_with_summary(tool: Tool) -> Tool:
    def wrapped_fn(input_str: str) -> dict:  # Expect a dict
        try:
            original_result = tool.invoke({"input_str": input_str})
            if isinstance(original_result, dict) and "output" in original_result:
                original_result = original_result["output"]
            original_result = clean_surrogates(str(original_result))
        except Exception as e:
            return {"final_answer": f"Tool execution failed: {e}"}  # Return error in dict

        prompt = summary_prompt.format(user_question=input_str, tool_result=original_result)
        summary = llm.invoke(prompt).content
        cleaned_summary = clean_surrogates(summary)
        return {"final_answer": cleaned_summary}  # Return final answer in dict

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


# Prompt Templates
# agents.py
tool_selection_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are an expert at selecting the right tool to answer the user's question.
You will be given a user question and a list of available tools.
Your job is to select the single best tool that can answer the question.
Respond with the tool name EXACTLY as it is written, or "NONE" if no tool is appropriate.

After the tool provides its output, you MUST formulate a concise and complete final answer to the user's question, using the tool's result.

Available Tools:
{tool_descriptions}
"""),
    HumanMessage(content="{user_question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

general_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful and informative chat assistant."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

default_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart agent helping with dataset analysis."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# LLM-Based Tool Selection Agent
tool_selection_agent = create_tool_calling_agent(
    llm=llm,
    tools=all_tools,  # Give it all tools to choose from
    prompt=tool_selection_prompt.partial(tool_descriptions="\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools]))
)
tool_selection_executor = AgentExecutor(
    agent=tool_selection_agent,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True
)

# General Chat Agent
general_chat_agent = create_tool_calling_agent(
    llm=llm,
    tools=[],  # General chat doesn't use tools
    prompt=general_chat_prompt
)
general_chat_executor = AgentExecutor(
    agent=general_chat_agent,
    tools=[],
    verbose=True,
    handle_parsing_errors=True
)

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

    print("\n--- Agent Routing Debugging ---")  # Start of routing log
    print(f"User Message: {user_message}")

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
    regex_tool_map = [
        (r'\b(linear regression|simple regression|regression equation)\b', linear_regression_tool),
        (r'\b(polynomial regression|curve fitting|nonlinear regression)\b', polynomial_regression_tool),
        (r'\b(preprocess|clean|missing values|data cleaning)\b', preprocess_tool),
        (r'\b(correlation|pearson|spearman|relationship|relation)\b', correlation_tool),
        (r'\b(predict|forecast|future estimate|prediction)\b', get_dynamic_prediction_tool),
        (r'\b(insight|reporting|report)\b', get_report_generator_tool),
        (r'\b(average|total|highest|maximum|min|lowest|entries|count|how many|mean)\b', get_history_tool),
        (r'\b(classification|classify|ann|neural network)\b', get_ann_tool),
        (r'\b(summary|summarize)\b', get_simple_summary_tool),
    ]
    for pattern, tool in regex_tool_map:
        if re.search(pattern, message):
            print(f"  - Regex matched tool: {tool.name}")
            return AgentExecutor(agent=create_tool_calling_agent(llm=llm, tools=[tool], prompt=default_prompt),
                                 tools=[tool], verbose=True, handle_parsing_errors=True), user_message

    # 2. LLM-Based Tool Selection (Fallback)
    print("  2. No Regex match. Invoking LLM Tool Selection Agent...")
    tool_result = tool_selection_executor.invoke({"user_question": user_message})
    tool_name = tool_result["output"]
    if tool_name != "NONE":
        selected_tool = next((tool for tool in all_tools if tool.name == tool_name), None)
        if selected_tool:
            tool_input = {"input_str": user_message}  # Or construct the correct input
            print(f"  - Calling tool '{selected_tool.name}' with input: {tool_input}")
            print("  - LLM selected a tool. Executing it.")
            return AgentExecutor(
                agent=create_tool_calling_agent(llm=llm, tools=[selected_tool], prompt=default_prompt),
                tools=[selected_tool],
                verbose=True,
                handle_parsing_errors=True
            ), user_message

    # 3. General Chat Agent (Final Fallback)
    print("  3. No tool selected by Regex or LLM. Routing to General Chat.")
    return general_chat_executor, user_message