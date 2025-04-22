import os
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
from langchain.agents import tool, initialize_agent, AgentType, Tool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from scipy.stats import spearmanr, pearsonr

# ===== Tool Functions =====

def get_column_names(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return f"Columns: {', '.join(df.columns)}"

def perform_linear_regression(file_path: str, target: str, features) -> dict:
    df = pd.read_csv(file_path)
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",")]
    df = df[features + [target]].dropna()
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    intercept = model.intercept_
    coefs = model.coef_
    terms = " + ".join([f"{coef:.4f}*{feat}" for coef, feat in zip(coefs, features)])
    equation = f"{target} = {terms} + {intercept:.4f}"
    return {
        "equation": equation,
        "r2_score": r2
    }

@tool
def linear_regression_tool(input_str: str) -> str:
    """Perform linear regression on CSV. Format: 'file_path=<path>; target=<target>; features=<feat1,feat2,...>'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")
    try:
        result = perform_linear_regression(file_path, target, features)
        return f"Regression Equation: {result['equation']}\n R² Score: {result['r2_score']:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

def perform_polynomial_regression(file_path: str, target: str, features: list, degree: int) -> str:
    df = pd.read_csv(file_path)
    X = df[features].values
    y = df[target].values
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    score = model.score(X_poly, y)
    return f"Polynomial Regression (degree={degree}) done.\n📊 R² Score: {score:.4f}"

@tool
def polynomial_regression_tool(input_str: str) -> str:
    """Perform polynomial regression. Format: 'file_path=<path>; target=<target>; features=<feat1,feat2,...>; degree=<int>'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")
    degree = int(params.get("degree", "2").strip())
    try:
        return perform_polynomial_regression(file_path, target, features, degree)
    except Exception as e:
        return f" Error: {str(e)}"

def classify_with_ann(file_path: str, target: str, features: list) -> str:
    df = pd.read_csv(file_path)
    X = df[features].astype(np.float32).values
    y = df[target].values
    classes = list(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[val] for val in y]).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    class SimpleANN(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            )
        def forward(self, x):
            return self.fc(x)

    model = SimpleANN(X.shape[1], len(classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(torch.tensor(X_train))
        loss = loss_fn(out, torch.tensor(y_train))
        loss.backward()
        optimizer.step()

    model.eval()
    preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    acc = accuracy_score(y_test, preds)
    return f"ANN classification complete.\n🎯 Accuracy: {acc:.2f}"

@tool
def ann_classifier_tool(input_str: str) -> str:
    """Train a simple ANN on CSV. Format: 'file_path=<path>; target=<target>; features=<feat1,feat2,...>'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")
    try:
        return classify_with_ann(file_path, target, features)
    except Exception as e:
        return f"Error: {str(e)}"

def preprocess_dataset(csv_path: str) -> str:
    """
    Cleans and preprocesses a dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file to preprocess.

    Returns:
        str: A message indicating the completion of preprocessing.
    """
    try:
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be a string.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"Error: File not found at path: {csv_path}"
    except TypeError as te:
        return f"Error: Invalid input type: {te}"
    except Exception as e:
        return f"Error reading CSV: {e}"

    # Clean & preprocess (as before)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('', np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                try:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                except:
                    pass

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue

    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(col, axis=1, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].dropna().unique()
        if set(map(str.lower, map(str, unique_vals))) <= {'yes', 'no', 'true', 'false'}:
            df[col] = df[col].str.lower().map({'yes': True, 'no': False, 'true': True, 'false': False})

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', str(x)) if isinstance(x, str) else x)

    # Save cleaned file
    output_path = os.path.splitext(csv_path)[0] + "_cleaned.csv"
    df.to_csv(output_path, index=False)

    return f"Preprocessing complete. Cleaned file saved to: {output_path}"

preprocess_tool = Tool(
    name="Dataset Preprocessor",
    func=preprocess_dataset,
    description="Use this tool to clean and preprocess a dataset. Provide a CSV file path as input. It returns a message when done.",
)

@tool
def correlation_tool(input_str: str) -> str:
    """
    Extracts two column names from a user query and computes Pearson and Spearman correlation.
    Example query: 'Find correlation between sales and profit'
    """
    
    # Parse input to extract the file path and query
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    query = params.get("query", "").strip()

    if not file_path:
        return "Please provide a valid file path."

    try:
        # Load the DataFrame
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {str(e)}"

    # Extract columns based on the query
    def extract_columns(query, df_columns):
        found_columns = []
        for col in df_columns:
            if re.search(rf'\b{col}\b', query, re.IGNORECASE):
                found_columns.append(col)
        return found_columns if len(found_columns) == 2 else None

    # Calculate correlations
    def calculate_correlations(df, col1, col2):
        x = df[col1].dropna()
        y = df[col2].dropna()
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)
        return pearson_corr, spearman_corr

    # Find the columns to correlate
    columns = extract_columns(query, df.columns)

    if columns:
        col1, col2 = columns
        pearson_corr, spearman_corr = calculate_correlations(df, col1, col2)
        return (
            f"Pearson correlation between '{col1}' and '{col2}': {pearson_corr:.4f}\n"
            f"Spearman correlation between '{col1}' and '{col2}': {spearman_corr:.4f}"
        )
    else:
        return "Could not identify two valid columns from your query."



# ===== Intent Detection Function =====

import re
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate

def detect_intent(llm: BaseLanguageModel, user_input: str) -> str:
    """Uses LLM to detect intent and returns a simple label."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier. Your job is to classify the user's message into one of the following intents:

         - greeting: For salutations, introductions, or starting a conversation (e.g., "Hi", "Hello", "Good morning")
         - general_qa: For general questions seeking information (e.g., "What is the capital of France?", "Tell me about...")
         - summarize_tool: If the user wants a file summary
         - correlation_tool: If the user asks for correlation or relationships
         - linear_regression_tool: If the user wants predictions or linear modeling
         - polynomial_regression_tool: For complex predictions
         - ann_classification_tool: For classification tasks

         Return ONLY the intent name, nothing else.

         EXAMPLES:
         User Input: Hey there!
         Intent: greeting

         User Input: What time is it?
         Intent: general_qa

         User Input: Good evening, bot.
         Intent: greeting

         User Input: Explain the theory of relativity.
         Intent: general_qa"""),
        ("human", "{text}")
    ])
    chain = prompt | llm
    result = chain.invoke({"text": user_input})

    # 🛠 Fix: extract content string from LLM result
    if hasattr(result, "content"):
        return result.content.strip().lower()
    return str(result).strip().lower()


# ===== LangChain LLM + Agent =====

llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [
    linear_regression_tool,
    polynomial_regression_tool,
    ann_classifier_tool,
    preprocess_tool,
    correlation_tool,
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True 
)

# ===== Final Agent Call with Intent Detection =====

# chain_llm.py
def get_bot_response(user_id: str, user_input: str) -> str:
    """Detect user intent and get response from agent or LLM with user context"""
    try:
        # Step 1: Detect intent using LLM
        intent = detect_intent(llm, user_input)
        print(f"[DEBUG] Detected Intent: {intent}")

        # Step 2: Routing based on intent
        if intent in ["Greeting", "chitchat", "greeting", "unknown","general_qa"]:
            # General Q&A handled by LLM directly
            response = llm.invoke(user_input)
            content = response.content if hasattr(response, "content") else str(response)
            return f"**Detected Intent:** `{intent}`\n\n**Response:**\n{content}"

        # Step 3: Use tools for analysis-related intents
        if intent in ["summarize_tool", "correlation_tool", "linear_regression_tool", "polynomial_regression_tool", "ann_classification_tool"]:
            # Handle analysis queries by invoking tools
            response = agent.invoke({"input": user_input})
            content = response.get("output", str(response))
            return f"**Detected Intent:** `{intent}`\n\n**Response:**\n{content}"

        # Step 4: Secure code execution for file uploads
        if intent == "code_execution_tool":  # example for file-based tasks
            # Call your secure notebook code execution here
            response = handle_file_upload_and_execution(user_input)
            return f"**Detected Intent:** `{intent}`\n\n**Response:**\n{response}"

    except Exception as e:
        return f"**Detected Intent:** `error`\n\n**Response:**\n{str(e)}"


def handle_file_upload_and_execution(user_input: str):
    """Secure execution of code in a notebook environment for file-related tasks."""
    # You can create specific handling based on the user input, such as:
    # - Reading the uploaded file
    # - Processing or running code in a secure environment
    # - Returning results to the user

    # For example:
    file_path = extract_file_path(user_input)  # assuming you extract file path from input
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Process the file and execute any necessary code
        result = "File processed successfully."  # Example result
        return result
    else:
        return "Error: File not found."

