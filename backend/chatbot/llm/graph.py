# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from .agents import route_tool_by_intent
from .utils import parse_tool_input  # Add this import
import os
from django.conf import settings

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "List of conversation messages"]
    context: Annotated[Dict[str, Any], "Contextual information (e.g., uploaded files)"]

def validate_file_context(context: Dict[str, Any]) -> bool:
    """Validate that the file in context exists and is accessible"""
    if not context.get("file_id"):
        return False
        
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, 'user_uploads', str(context["file_id"]))
        return os.path.exists(file_path)
    except Exception:
        return False

def agent_router_node(state: AgentState):
    try:
        last_message = state["messages"][-1]
        user_message = last_message.content if isinstance(last_message.content, str) else last_message.content.get("user_message", "")
        
        # Input validation
        if not user_message or not isinstance(user_message, str):
            return {
                "messages": state["messages"] + [AIMessage(content="Please provide a valid query.")],
                "context": state["context"]
            }

        # Validate file context if present
        if state["context"].get("file_id") and not validate_file_context(state["context"]):
            return {
                "messages": state["messages"] + [AIMessage(content="The file reference is invalid. Please re-upload your file.")],
                "context": state["context"]
            }

        executor, tool_output = route_tool_by_intent(user_message, state["context"])
        
        # Format tool input properly
        tool_input = parse_tool_input(
            tool_output if isinstance(tool_output, str) else str(tool_output),
            {**state["context"], "user_message": user_message}
        )

        try:
            # Add timeout configuration
            config = {
                "run_name": "agent_router",
                "configurable": {
                    "timeout": 30  # 30 second timeout
                }
            }
            
            result = executor.invoke({
                "messages": [HumanMessage(content=tool_input)]
            }, config=config)

            bot_response = result.get("output", "Sorry, I couldn't understand that.")
            
            # Clean the response if needed
            if isinstance(bot_response, dict):
                bot_response = bot_response.get("final_answer", str(bot_response))
                
            return {
                "messages": state["messages"] + [AIMessage(content=bot_response)],
                "context": state["context"]
            }
            
        except TimeoutError:
            return {
                "messages": state["messages"] + [AIMessage(content="The operation timed out. Please try a simpler query or smaller dataset.")],
                "context": state["context"]
            }
        except Exception as e:
            print(f"Agent execution failed: {str(e)}")
            return {
                "messages": state["messages"] + [AIMessage(content="I encountered an error processing your request. Please try again.")],
                "context": state["context"]
            }
            
    except Exception as e:
        print(f"Router node failed: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="System error occurred. Please try again.")],
            "context": state["context"]
        }

graph = StateGraph(AgentState)
graph.add_node("agent_router", agent_router_node)
graph.set_entry_point("agent_router")
graph.add_edge("agent_router", END)
runnable_graph = graph.compile()