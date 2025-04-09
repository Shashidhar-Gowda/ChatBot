# rule_based_agent_pipeline.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
from google.cloud import bigquery
### ----------- INPUT HANDLER TOOL ------------ ###
@tool
def load_input(input_type: str, input_value: str) -> str:
    """
    Load data from CSV, JSON, or BigQuery.
    Args:
        input_type: 'csv', 'json', or 'bigquery'
        input_value: file path or query string
    Returns:
        path to cleaned temporary CSV file
    """
    try:
        if input_type == "csv":
            df = pd.read_csv(input_value)
        elif input_type == "json":
            df = pd.read_json(input_value)
        elif input_type == "bigquery":
            client = bigquery.Client()
            df = client.query(input_value).to_dataframe()
        else:
            return "Invalid input type."
        path = f"temp_{uuid4().hex}.csv"
        df.to_csv(path, index=False)
        return path
    except Exception as e:
        return f"Error loading input: {str(e)}"
### ----------- PREPROCESS TOOL ------------ ###
@tool
def preprocess_data(file_path: str) -> str:
    """
    Cleans the dataset: drops nulls, fixes types.
    Returns path to cleaned CSV.
    """
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        cleaned_path = f"cleaned_{uuid4().hex}.csv"
        df.to_csv(cleaned_path, index=False)
        return cleaned_path
    except Exception as e:
        return f"Error preprocessing: {str(e)}"
### ----------- VISUALIZATION TOOL ------------ ###
@tool
def generate_visualizations(file_path: str) -> str:
    """Creates histograms and pair plots, saves to 'viz_outputs/', returns list of image paths."""
    try:
        df = pd.read_csv(file_path)
        os.makedirs("viz_outputs", exist_ok=True)
        paths = []
        for col in df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.histplot(df[col], kde=True)
            path = f"viz_outputs/{col}_hist_{uuid4().hex}.png"
            plt.title(f"Distribution of {col}")
            plt.savefig(path)
            plt.close()
            paths.append(path)
        sns.pairplot(df.select_dtypes(include=['number']))
        pair_path = f"viz_outputs/pairplot_{uuid4().hex}.png"
        plt.savefig(pair_path)
        plt.close()
        paths.append(pair_path)
        return json.dumps(paths)
    except Exception as e:
        return f"Error generating visualizations: {str(e)}"
### ----------- CORRELATION TOOL ------------ ###
@tool
def correlation_analysis(file_path: str) -> str:
    """
    Returns Pearson and Spearman correlation matrices.
    """
    try:
        df = pd.read_csv("/content/assignment_data.csv")
        pearson = df.corr(method="pearson").round(3).to_dict()
        spearman = df.corr(method="spearman").round(3).to_dict()
        return json.dumps({"pearson": pearson, "spearman": spearman}, indent=2)
    except Exception as e:
        return f"Error in correlation analysis: {str(e)}"
### ----------- RULE-BASED AGENT ------------ ###
def run_rule_based_agent(input_type: str, input_value: str):
    tools = [load_input, preprocess_data, generate_visualizations, correlation_analysis]
    llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")
    agent = create_tool_calling_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    query = f"""
    1. Load the dataset from {input_type} source.
    2. Preprocess the dataset.
    3. Generate visualizations.
    4. Perform Pearson and Spearman correlation analysis.
    """
    return agent_executor.invoke({"input": query, "input_type": input_type, "input_value": input_value})
