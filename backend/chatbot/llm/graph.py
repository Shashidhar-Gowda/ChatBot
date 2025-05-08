# graph.py
import re
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from .agents import route_tool_by_intent
from .intent_detector import detect_intent


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "List of conversation messages"]
    context: Annotated[Dict[str, Any], "Contextual information (e.g., uploaded files)"]


def clean_think_messages(text: str) -> str:
    """Removes <think>...</think> blocks and tool traces from a string."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned_text = re.sub(r"Invoking: .*?\n", "", cleaned_text).strip()  # Remove "Invoking: ..." lines
    cleaned_text = re.sub(r"> Finished chain\.\n", "", cleaned_text).strip()  # Remove "> Finished chain."
    cleaned_text = re.sub(r"Entering new AgentExecutor chain\.\.\.\n", "", cleaned_text).strip() # Remove chain enter log
    return cleaned_text


def agent_router_node(state: AgentState):
    user_message = state["messages"][-1].content
    intent = detect_intent(user_message)
    context = state["context"]
    context["intent"] = intent
    executor, updated_user_message = route_tool_by_intent(user_message, context)

    result = executor.invoke({
        "messages": [HumanMessage(content=updated_user_message)]
    })

    if isinstance(result, dict) and "final_answer" in result:
        bot_response = result["final_answer"]
    else:
        bot_response = result.get("output", "Sorry, I couldn't understand that.")

    # --- MODIFIED: Clean the bot's response ---
    bot_response = result.get("output", "Sorry, I couldn't understand that.")
    cleaned_response = clean_think_messages(bot_response)
    ai_message = AIMessage(content=cleaned_response)
    # --- END OF MODIFICATION ---

    return {"messages": state["messages"] + [ai_message], "context": context}


graph = StateGraph(AgentState)
graph.add_node("agent_router", agent_router_node)
graph.set_entry_point("agent_router")
graph.add_edge("agent_router", END)
runnable_graph = graph.compile()