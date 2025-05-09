# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from .agents import route_tool_by_intent

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "List of conversation messages"]
    context: Annotated[Dict[str, Any], "Contextual information (e.g., uploaded files)"]  # Add context

def agent_router_node(state: AgentState):
    last_message_content = state["messages"][-1].content

    # Safely extract user_message, handling both string and dict
    user_message = last_message_content if isinstance(last_message_content, str) else last_message_content.get("user_message", "")

    executor, tool_output = route_tool_by_intent(user_message, state["context"])

    # Safely extract user_message for HumanMessage, handling both string and dict
    next_message_content = tool_output.get("user_message", "") if isinstance(tool_output, dict) else str(tool_output)

    result = executor.invoke({
        "messages": [HumanMessage(content=next_message_content)]
    })

    bot_response = result.get("output", "Sorry, I couldn't understand that.")
    ai_message = AIMessage(content=bot_response)

    return {"messages": state["messages"] + [ai_message], "context": state["context"]}


graph = StateGraph(AgentState)
graph.add_node("agent_router", agent_router_node)
graph.set_entry_point("agent_router")
graph.add_edge("agent_router", END)
runnable_graph = graph.compile()