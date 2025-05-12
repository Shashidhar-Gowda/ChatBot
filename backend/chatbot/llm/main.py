import pandas as pd
import re
from pathlib import Path
from django.conf import settings
from langchain_core.messages import HumanMessage, AIMessage
from .intent_detector import detect_intent
from .utils import resolve_file_path  # Import the path resolver
from .graph import runnable_graph


from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Initialize chat_history as a module-level variable
chat_history: List[BaseMessage] = []

def clean_think_messages(text: str) -> str:
    """Removes <think>...</think> blocks and tool traces from a string."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned_text = re.sub(r"Invoking: .*?\n", "", cleaned_text).strip()
    cleaned_text = re.sub(r"> Finished chain\.\n", "", cleaned_text).strip()
    cleaned_text = re.sub(r"Entering new AgentExecutor chain\.\.\.\n", "", cleaned_text).strip()
    return cleaned_text

def get_bot_response(user_input: str, file_name: str = None) -> Dict:
    """
    Handles user input with optional file reference.
    """
    global chat_history  # Declare we're using the global chat_history
    
    try:
        # 1. Detect intent
        intent = detect_intent(user_input)
        
        # 2. Prepare context
        context = {}
        if file_name:
            try:
                # Validate file exists
                file_path = resolve_file_path(file_name)
                context = {
                    "file_id": file_name,
                    "uploaded_files": {file_name: file_path}
                }
                
                # If user asks about columns, handle it directly
                if "column" in user_input.lower() or "field" in user_input.lower():
                    df = pd.read_csv(file_path)
                    cols = df.columns.tolist()
                    return {
                        "intent": "COLUMN_INFO",
                        "bot_reply": f"The dataset contains these columns: {', '.join(cols)}"
                    }
                    
            except Exception as e:
                return {
                    "intent": "ERROR",
                    "bot_reply": f"File error: {str(e)}"
                }
        
        # 3. Add user message to history
        chat_history.append(HumanMessage(content=user_input))
        
        # 4. Run the agent
        response = runnable_graph.invoke({
            "messages": chat_history,
            "context": context
        })
        
        # 5. Get and clean bot reply
        bot_reply = response["messages"][-1].content if response.get("messages") else "No response."
        cleaned_bot_reply = clean_think_messages(bot_reply)
        
        # 6. Add bot reply to history
        chat_history.append(AIMessage(content=cleaned_bot_reply))
        
        return {
            "intent": intent,
            "bot_reply": cleaned_bot_reply
        }
        
    except Exception as e:
        print(f"Error in get_bot_response: {e}")
        return {
            "intent": "ERROR",
            "bot_reply": "Sorry, I encountered an error processing your request."
        }

def reset_chat_history():
    """Reset the conversation history"""
    global chat_history
    chat_history = []