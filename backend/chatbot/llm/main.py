# main.py
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from .graph import runnable_graph
import os
from ..models import UploadedFile
from django.core.files.storage import default_storage

chat_history = []
uploaded_files = {}  # Dictionary to store uploaded file paths

def get_latest_uploaded_file(user):
    return UploadedFile.objects.filter(user=user).order_by('-uploaded_at').first()


def get_bot_response(user_input: str, file_id: str = None, user=None) -> str:
    global chat_history
    chat_history.append(HumanMessage(content=user_input))

    context = {}

    # ✅ Use latest file if file_id is not provided
    if not file_id and user:
        latest_file = get_latest_uploaded_file(user)
        if latest_file:
            file_id = str(latest_file.id)

    if file_id:
        try:
            uploaded_instance = UploadedFile.objects.get(id=file_id)
            full_file_path = default_storage.path(uploaded_instance.file.name)
            context["uploaded_files"] = {uploaded_instance.filename: full_file_path}
            context["file_id"] = str(file_id)
        except UploadedFile.DoesNotExist:
            context["uploaded_files"] = {}

    inputs = {
        "messages": chat_history,
        "context": context
    }

    response = runnable_graph.invoke(inputs)

    if "messages" in response and len(response["messages"]) > 1:
        bot_reply = response["messages"][-1].content
    else:
        bot_reply = "OK. File uploaded. What would you like to analyze?"

    chat_history.append(AIMessage(content=bot_reply))
    return bot_reply




if __name__ == "__main__":
    print("\n🤖 Welcome to Smart Data Analyst Bot!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break

        bot_response = get_bot_response(user_input)
        print(f"Bot: {bot_response}\n")