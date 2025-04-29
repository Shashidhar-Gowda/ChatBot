
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.stats import pearsonr, spearmanr
from difflib import get_close_matches
from .utils import get_sql_database
from langchain.tools import tool, Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from io import StringIO  
from sqlalchemy import text 
from langchain_core.runnables import RunnableConfig

# Instantiate LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ============================== BASE FUNCTIONS ==============================

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
    return {"equation": equation, "r2_score": r2}

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

def preprocess_dataset(csv_path: str) -> str:
    try:
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be a string.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}")
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}"

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

    output_path = os.path.splitext(csv_path)[0] + "_cleaned.csv"
    df.to_csv(output_path, index=False)

    return f"Preprocessing complete. Cleaned file saved to: {output_path}"

def predict_future(df, target_column, feature_columns):
    df = df.dropna(subset=[target_column] + feature_columns)
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {"model": model, "mse": mse, "prediction_example": preds[:5]}

def generate_insights(df):
    prompt = PromptTemplate(
        input_variables=["data"],
        template="Given the following dataset snippet, generate a brief analysis report with insights:\n\n{data}"
    )
    data_preview = df.head(10).to_string()
    llm_prompt = prompt.format(data=data_preview)
    messages = [HumanMessage(content=llm_prompt)]
    return llm(messages)

def basic_eda(df):
    return {
        "head": df.head(),
        "description": df.describe(),
        "missing_values": df.isnull().sum(),
        "correlation": df.corr(numeric_only=True)
    }

def generate_charts(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# ============================== TOOL WRAPPERS ==============================

from django.core.files.storage import default_storage
from ..models import UploadedFile  # Import your Django model

@tool
def analyze_uploaded_file(input_str: str) -> str: 
    """Analyze an uploaded file given its file_id."""
    file_id = input_str.strip()  # The input_str is expected to be the file_id

    if not file_id:
        return "Please provide a file ID."

    try:
        uploaded_file_instance = UploadedFile.objects.get(id=file_id) 
        file_path = default_storage.path(uploaded_file_instance.file.name)  
        df = pd.read_csv(file_path)
        return f"Analysis results:\n\n{df.head().to_markdown(index=False, numalign='left', stralign='left')}"
    except UploadedFile.DoesNotExist:
        return f"File not found with ID: {file_id}"
    except Exception as e:
        return f"Error analyzing file: {e}"

def linear_regression_tool_fn(input_str: str, run_config=None) -> str:
    """
    Perform linear regression. Format: 'target=target_column; features=feat1,feat2,...'
    Automatically uses the latest uploaded file from context if file_path is not provided.
    """
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()

    # 🧠 Auto-resolve file path from context
    if run_config and 'context' in run_config and 'uploaded_files' in run_config['context']:
        uploaded = list(run_config['context']['uploaded_files'].values())
        if uploaded:
            file_path = uploaded[0]

    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")

    try:
        result = perform_linear_regression(file_path, target, features)
        return f"Regression Equation: {result['equation']}\nR² Score: {result['r2_score']:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

from langchain_core.tools import Tool

linear_regression_tool = Tool.from_function(
    func=linear_regression_tool_fn,
    name="linear_regression_tool",
    description="Perform linear regression using a CSV file. Format: 'target=col; features=feat1,feat2,...'. Automatically uses the latest uploaded file if no file_path is specified.",
    infer_schema=True
)



@tool
def polynomial_regression_tool(input_str: str) -> str:
    """Perform polynomial regression. Format: 'file_path=path; target=target; features=feat1,feat2,...; degree=int'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")
    degree = int(params.get("degree", "2").strip())
    try:
        return perform_polynomial_regression(file_path, target, features, degree)
    except Exception as e:
        return f"Error: {str(e)}"

preprocess_tool = Tool(
    name="Dataset Preprocessor",
    func=preprocess_dataset,
    description="Cleans and preprocesses a dataset. Provide a CSV file path."
)

@tool
def correlation_tool(input_str: str) -> str:
    """Finds Pearson and Spearman correlation between two columns from a query."""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    query = params.get("query", "").strip()
    if not file_path:
        return "Please provide a valid file path."
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {str(e)}"

    def extract_columns(query, df_columns):
        found = []
        for col in df_columns:
            if re.search(rf'\b{col}\b', query, re.IGNORECASE):
                found.append(col)
        return found if len(found) == 2 else None

    def calculate_correlations(df, col1, col2):
        x = df[col1].dropna()
        y = df[col2].dropna()
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        return pearsonr(x, y)[0], spearmanr(x, y)[0]

    columns = extract_columns(query, df.columns)
    if columns:
        col1, col2 = columns
        pearson_corr, spearman_corr = calculate_correlations(df, col1, col2)
        return f"Pearson: {pearson_corr:.4f}\nSpearman: {spearman_corr:.4f}"
    else:
        return "Couldn't identify two columns."

@tool
def prediction_tool(input_str: str) -> str:
    """Builds a prediction model for future estimates."""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {str(e)}"
    return handle_prediction_questions(params.get("query", ""), df)

def handle_prediction_questions(query: str, df: pd.DataFrame) -> str:
    query_lower = query.lower()
    columns = df.columns.tolist()
    if not any(word in query_lower for word in ["predict", "forecast", "estimate", "future"]):
        return "This doesn't look like a prediction query."
    mentioned = [col for col in columns if col.lower() in query_lower]
    if not mentioned:
        return "Couldn't detect a target column."
    target = mentioned[-1]
    features = [col for col in columns if col != target]
    try:
        result = predict_future(df, target, features)
        preds = ", ".join([f"{p:.2f}" for p in result["prediction_example"]])
        return f"✅ Model trained to predict `{target}`\n📊 MSE: {result['mse']:.2f}\nPredictions: {preds}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def insight_tool(input_str: str) -> str:
    """Generate high-level insights from a dataset."""
    try:
        df = pd.read_csv(input_str)
        insights = generate_insights(df)
        return insights.content
    except Exception as e:
        return f"Error: {str(e)}"

#  ============================== NEW: ANN CLASSIFIER ==============================

@tool
def ann_classification_tool(input_str: str) -> str:
    """Classify data using ANN. Format: 'file_path=path; target=target'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_path = params.get("file_path", "").strip()
    target = params.get("target", "").strip()

    if not file_path or not target:
        return "Please provide file_path and target."

    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).dropna()
        y = df[target].dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return f"✅ ANN Classification Model built.\n🔍 Accuracy: {score:.2%}"
    except Exception as e:
        return f"Error: {str(e)}"