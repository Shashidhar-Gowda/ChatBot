# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from .agents import route_tool_by_intent

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "List of conversation messages"]
    context: Annotated[Dict[str, Any], "Contextual information (e.g., uploaded files)"]  # Add context

def agent_router_node(state: AgentState):
    user_message = state["messages"][-1].content
    executor, updated_user_message = route_tool_by_intent(user_message, state["context"])

    result = executor.invoke({
        "messages": [HumanMessage(content=updated_user_message)]
    })

    bot_response = result.get("output", "Sorry, I couldn't understand that.")
    ai_message = AIMessage(content=bot_response)

    return {"messages": state["messages"] + [ai_message], "context": state["context"]}


graph = StateGraph(AgentState)
graph.add_node("agent_router", agent_router_node)
graph.set_entry_point("agent_router")
graph.add_edge("agent_router", END)
runnable_graph = graph.compile()