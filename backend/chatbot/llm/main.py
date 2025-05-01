# main.py

import os
from langchain.schema import HumanMessage, AIMessage
from .graph import runnable_graph
from .intent_detector import detect_intent
from .column_matcher import match_columns
# from .utils import get_latest_file_for_user

chat_history = []

def get_bot_response(user_input: str, file_id: int = None) -> dict:
    """
    Handles user input: detects intent, matches columns, runs LangChain agent,
    and returns a dict containing all relevant info for frontend or CLI.
    """
    global chat_history

    # 1. Detect intent
    intent = detect_intent(user_input)

    # 2. Match columns using actual file
    # matched_cols = []
    # if file_id:
    #     try:
    #         df = get_latest_file_for_user(file_id)
    #         matched_cols = match_columns(df.columns.tolist(), user_input)
    #     except Exception as e:
    #         print(f"Error loading file for column matching: {e}")

    # # 3. Optionally enhance the user prompt
    # cleaned_query = user_input
    # if matched_cols and matched_cols[0].lower() != "none":
    #     for col in matched_cols:
    #         if col.lower() not in user_input.lower():
    #             cleaned_query += f" (column: {col})"

    # 4. Prepare context for agent
    context = {"file_id": str(file_id)}
    if file_id:
        # Also pass dummy uploaded_files mapping if needed by graph
        context["uploaded_files"] = {str(file_id): os.path.join('media', str(file_id))}

    # 5. Run LangChain graph
    chat_history.append(HumanMessage(content=user_input))
    response = runnable_graph.invoke({
        "messages": chat_history,
        "context": context
    })
    bot_reply = response["messages"][-1].content if response.get("messages") else "No response."

    chat_history.append(AIMessage(content=bot_reply))

    return {
        "intent": intent,
        # "matched_columns": matched_cols,
        # "query_used": cleaned_query,
        "bot_reply": bot_reply
    }


