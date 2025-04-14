import os
import re

def extract_intent(llm_output):
    """
    Extracts the intent label from LLM output by removing <think> tags
    Args:
        llm_output: Raw string output from LLM
    Returns:
        Cleaned intent label string
    """
    # Remove everything in <think>...</think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL)
    # Remove any remaining HTML-style tags and whitespace
    cleaned = re.sub(r'<[^>]+>', '', cleaned).strip()
    return cleaned

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
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Groq initialization failed: {str(e)}")
    llm = None

# System prompt
system_prompt = """You are BrainBot, an expert data analysis assistant. Your capabilities include:
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

def detect_intent(prompt: str) -> str:
    """Detect intent from user prompt using LLM with more specific categories"""
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier. 
        Classify the user's input into a concise, specific intent category. 
        Respond with ONLY the intent label — a single word or at most two words that best capture the user's intent.
        Do not include explanations or any other text.
        Just return the intent label."""),
        ("human", "{query}")
    ])
    
    try:
        chain = intent_prompt | llm | StrOutputParser()
        result = chain.invoke({"query": prompt})
        # Clean the result using the extract_intent function
        return extract_intent(result).lower()
    except Exception as e:
        print(f"Intent detection error: {str(e)}")
        return "error"

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

def get_bot_response(user_id: str, user_input: str, context: str = None) -> dict:
    """Enhanced response handler that returns standardized response structure"""
    # First detect intent regardless of query type
    intent = detect_intent(user_input)
    
    # Handle name queries first
    name_response = handle_name_queries(user_id, user_input)
    if name_response:
            return {
                "metadata": {
                    "intent": intent,
                    "detected_at": datetime.now().isoformat(),
                    "confidence": "high"
                },
                "response": name_response,
                "follow_up": [],
                "type": "dict",
                "is_dict": True
            }
        
    if not agent_executor:
        return {
            "metadata": {
                "intent": intent,
                "detected_at": datetime.now().isoformat(),
                "confidence": "high"
            },
            "response": "System error: Chat functionality is currently unavailable",
            "follow_up": [],
            "type": "dict",
            "is_dict": True
        }
    
    # Check if this is a general question (not data/analysis related)
    is_general_question = not any(term in user_input.lower() for term in [
        'data', 'analyze', 'stat', 'model', 'predict', 'correlate', 
        'regress', 'classify', 'visualize', 'chart', 'graph'
    ])
    
    if is_general_question or len(user_input.split()) > 15:
        try:
            prompt = f"User ({user_id}) asked: {user_input}"
            if context:
                prompt = f"{context}\n{prompt}"
            return {
                "intent": intent,
                "response": llm.invoke(prompt).content,
                "follow_up": generate_follow_up(intent),
                "type": "dict",
                "is_dict": True
            }
        except Exception as e:
            return {
                "intent": "error",
                "response": f"Error answering general question: {str(e)}",
                "follow_up": [],
                "type": "dict",
                "is_dict": True
            }
    
    # Otherwise proceed with tool-based analysis
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
        
        return {
            "intent": intent,
            "response": output,
            "follow_up": generate_follow_up(intent),
            "type": "dict",
            "is_dict": True
        }
    except Exception as e:
        if "no tool applicable" in str(e).lower():
            try:
                return {
                    "intent": intent,
                    "response": llm.invoke(user_input).content,
                    "follow_up": generate_follow_up(intent),
                    "type": "dict",
                    "is_dict": True
                }
            except Exception as llm_error:
                return {
                    "intent": "error",
                    "response": f"Error: {str(llm_error)}",
                    "follow_up": [],
                    "type": "dict",
                    "is_dict": True
                }
        return {
            "intent": "error",
            "response": f"Error processing request: {str(e)}",
            "follow_up": [],
            "type": "dict",
            "is_dict": True
        }

def generate_follow_up(intent: str) -> list:
    """Generate follow-up questions based on detected intent"""
    follow_ups = {
        "data_analysis": [
            "Would you like me to analyze specific columns?",
            "Should I generate visualizations for this data?",
            "Would you like statistical summaries?"
        ],
        "correlation_analysis": [
            "Which two columns should I analyze?",
            "Would you like a scatter plot of the correlation?",
            "Should I calculate the correlation matrix?"
        ],
        "regression_analysis": [
            "Which variable should I predict?",
            "Should I include all features in the model?",
            "Would you like performance metrics?"
        ],
        "general_question": [
            "Would you like more details?",
            "Should I explain this in simpler terms?",
            "Would you like related information?"
        ]
    }
    return follow_ups.get(intent, [])
