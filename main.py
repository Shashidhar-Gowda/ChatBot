import os
import pandas as pd
from dotenv import load_dotenv
from intent_detection import detect_intent
from agent_executor import AgentExecutorHandler

# Load environment variables
load_dotenv()

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        return "Unsupported file type."

def main():
    chat_history = []  # Store message history
    agent_executor = AgentExecutorHandler()

    dataset_path = input("Please enter the dataset file path (CSV/JSON): ")
    df = load_dataset(dataset_path)

    if isinstance(df, str):  # Error in dataset loading
        print(f"Error: {df}")
        return

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # Detect intent
        intent = detect_intent(query)
        print(f"[Detected Intent]: {intent}")

        # Use the agent to process the intent
        agent_output = agent_executor.invoke_agent(intent, df)

        # Log the query and response to chat history
        chat_history.append({"user": query, "response": agent_output})

        # Print the result (Tool output)
        print(f"Nuclei: {agent_output}")

if __name__ == "__main__":
    main()
