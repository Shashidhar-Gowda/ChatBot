from langchain_groq import ChatGroq
from datetime import datetime
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
try:
    from endpoints.tools import AnalysisTools
except ImportError:
    from .tools import AnalysisTools
from .mongo_db import get_user_memory, remember_name, recall_name
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq model with error handling
try:
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Groq initialization failed: {str(e)}")
    llm = None

# System prompt for DataSage
system_prompt = """You are DataSage, an expert data analysis assistant. Your capabilities include:
1. Data analysis (descriptive stats, correlations, regression, classification)
2. Data visualization (histograms, scatter plots, decision trees)
3. Statistical modeling and interpretation
4. Intelligent tool selection based on user queries"""

# Create tools
tools = [
    Tool(
        name="Data Analysis",
        func=AnalysisTools.describe_data,
        description="For performing basic descriptive statistics on data"
    ),
    Tool(
        name="Correlation Analysis",
        func=AnalysisTools.correlation_analysis,
        description="For calculating correlation between two columns in a dataset"
    ),
    Tool(
        name="Regression Analysis",
        func=AnalysisTools.linear_regression,
        description="For performing linear regression analysis on data"
    ),
    Tool(
        name="Classification",
        func=AnalysisTools.classify_data,
        description="For classifying data using simple artificial neural networks"
    )
]

# Enhanced agent configuration
if llm:
    tool_names = [tool.name for tool in tools]
    agent_prompt = hub.pull("hwchase17/react").partial(
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join(tool_names)
    )
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
        return_intermediate_steps=True
    )
else:
    agent_executor = None
    print("Warning: No LLM available - agent functionality disabled")

def detect_analysis_intent(prompt: str) -> str:
    """Detect analysis intent from user prompt"""
    prompt = prompt.lower()
    if any(term in prompt for term in ['statistic', 'summary', 'describe']):
        return 'data_analysis'
    elif any(term in prompt for term in ['correlate', 'relationship']):
        return 'correlation_analysis'
    elif any(term in prompt for term in ['predict', 'forecast', 'trend']):
        return 'regression_analysis'
    return 'unknown'

def load_file_data(file_path: str) -> pd.DataFrame:
    """Load file data based on extension"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    raise ValueError("Unsupported file format")

def handle_name_queries(user_id: str, user_input: str) -> Optional[str]:
    """Handle all name-related queries"""
    # Name setting
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1]
        name = name.split(".")[0].split("!")[0].strip()
        return remember_name(user_id, name)
        
    # Name recall
    name_phrases = [
        "what is my name",
        "what's my name", 
        "do you know my name",
        "did i tell you my name"
    ]
    if any(phrase in user_input.lower() for phrase in name_phrases):
        name = recall_name(user_id)
        return f"Your name is {name}" if name else "I don't know your name yet"
        
    return None

def reset_memory():
    """Reset the in-memory conversation history"""
    from .mongo_db import conversation_history
    conversation_history.clear()
    return {'status': 'success', 'message': 'Memory reset'}

def get_bot_response(user_id: str, user_input: str) -> str:
    """Enhanced response handler with dynamic tool selection"""
    # First check if this is a name-related query
    name_response = handle_name_queries(user_id, user_input)
    if name_response:
        return name_response
        
    if not agent_executor:
        return "System error: Chat functionality is currently unavailable"
    
    try:
        response = agent_executor.invoke({"input": user_input})
        output = response["output"]
        
        if isinstance(output, dict):
            if output.get("status") == "input_required":
                formatted = f"{output['message']}"
                if output.get("available_columns"):
                    formatted += f"\nAvailable columns: {', '.join(output['available_columns'])}"
                output = formatted
            elif output.get("status") == "error":
                output = f"Error: {output['message']}"
        
        return output
    except Exception as e:
        if "no tool applicable" in str(e).lower():
            try:
                return llm.invoke(user_input).content
            except Exception as llm_error:
                return f"Error: {str(llm_error)}"
        return f"Error processing request: {str(e)}"
