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
from .intent_detector import detect_intent

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

When you select a tool, respond with ONLY the following JSON structure:

```json
{
"tool_name": "the_tool_name",
"tool_input": {
"argument1": "value1",
"argument2": "value2"
}
}
```

Replace "the_tool_name" with the exact name of the tool.
Replace "argument1", "argument2", etc., with the exact names of the arguments
that the tool expects. Provide valid values for the arguments.

If no tool is appropriate to answer the user's question, respond with ONLY:

```json
{
"tool_name": "NONE"
}
```

Do not include any other text or explanations in your response.

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

    print("\n--- Agent Routing Debugging ---")  # Start of routing log
    print(f"User Message: {user_message}")

    # Intent Detection
    intent = detect_intent(user_message)
    print(f"  - Detected Intent: {intent}")

    # Short-circuit for simple intents
    if intent in ["GREETINGS", "GENERAL QUESTIONS"]:
        print("  - Directly routing to General Chat Agent for greeting or general question.")
        return general_chat_executor, user_message
    
    

    # Try matching uploaded file names
    if uploaded_files:
        # Exact match first
        for filename, file_id in uploaded_files.items():
            if filename.lower() in message:
                context["file_id"] = str(file_id)
                user_message = user_message.replace(filename, uploaded_files[filename])
                break

    # Regex Tool Matching
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

    # LLM-Based Tool Selection (Fallback)
    print("  - No Regex match. Invoking LLM Tool Selection Agent...")
    tool_result = tool_selection_executor.invoke({"user_question": user_message})
    tool_name = tool_result["output"]
    if tool_name != "NONE":
        selected_tool = next((tool for tool in all_tools if tool.name == tool_name), None)
        if selected_tool:
            print(f"  - LLM selected tool: {selected_tool.name}")
            return AgentExecutor(
                agent=create_tool_calling_agent(llm=llm, tools=[selected_tool], prompt=default_prompt),
                tools=[selected_tool],
                verbose=True,
                handle_parsing_errors=True
            ), user_message

    # Final Fallback to General Chat
    print("  - No tool selected by Regex or LLM. Routing to General Chat.")
    return general_chat_executor, user_message
