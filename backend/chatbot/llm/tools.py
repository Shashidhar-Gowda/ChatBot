
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
from .utils import resolve_file_path
from sklearn.preprocessing import StandardScaler
from langchain.chains import LLMChain
from typing import Callable  # Import Callable for type hinting
from typing import Optional  
from django.core.files.storage import default_storage
from ..models import UploadedFile  # Import your Django model
from typing import Union
from sklearn.metrics import accuracy_score


# Instantiate LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ============================== BASE FUNCTIONS ==============================


def get_column_names(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return f"Columns: {', '.join(df.columns)}"

# ===========================================================================================

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

@tool
def linear_regression_tool(input_str: str, request) -> str:  # Add request
    """Perform linear regression. Format: 'file_path=filename.csv; target=target; features=feat1,feat2,...'"""
    import re
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")

    try:
        real_path = resolve_file_path(file_name)  # Pass request.user
        result = perform_linear_regression(real_path, target, features)
        return f"Regression Equation: {result['equation']}\n R² Score: {result['r2_score']:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

# ===========================================================================================

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
    """Perform polynomial regression. Format: 'file_path=path; target=target; features=feat1,feat2,...; degree=int'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").split(",")
    degree = int(params.get("degree", "2").strip())
    try:
        real_path = resolve_file_path(file_name)
        return perform_polynomial_regression(real_path, target, features, degree)
    except Exception as e:
        return f"Error: {str(e)}"
    
# ===========================================================================================

def preprocess_dataset(file_name: str) -> str:
    try:
        if not isinstance(file_name, str):
            raise TypeError("csv_path must be a string.")
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found at path: {file_name}")
        df = pd.read_csv(file_name)
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

    output_path = os.path.splitext(file_name)[0] + "_cleaned.csv"
    df.to_csv(output_path, index=False)

    return f"Preprocessing complete. Cleaned file saved to: {output_path}"

from langchain.tools import tool

@tool
def preprocess_tool(file_name: str) -> str:
    """
    Cleans and preprocesses a dataset. Provide the CSV file name (not path).
    Returns the path to the cleaned CSV file.
    """
    try:
        real_path = resolve_file_path(file_name)
        return preprocess_dataset(real_path)
    except Exception as e:
        return f"❌ Failed to preprocess dataset: {e}"


# ===========================================================================================

def interpret_correlation(value: float) -> str:
    if abs(value) < 0.1:
        return "negligible correlation"
    elif abs(value) < 0.3:
        return "weak correlation"
    elif abs(value) < 0.5:
        return "moderate correlation"
    elif abs(value) < 0.7:
        return "strong correlation"
    else:
        return "very strong correlation"

def perform_correlation_analysis(file_path: str, target: str, features: Union[list, str]) -> dict:
    df = pd.read_csv(file_path)
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",")]
    df = df[features + [target]].dropna()
    
    results = {}
    for feat in features:
        pearson_corr, _ = pearsonr(df[feat], df[target])
        spearman_corr, _ = spearmanr(df[feat], df[target])
        results[feat] = {
            "pearson": round(pearson_corr, 4),
            "spearman": round(spearman_corr, 4),
            "pearson_interpretation": interpret_correlation(pearson_corr),
            "spearman_interpretation": interpret_correlation(spearman_corr)
        }
    return results

@tool
def correlation_tool(input_str: str) -> str:
    """Perform correlation analysis. Format: 'file_path=path; target=target; features=feat1,feat2,...'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = params.get("features", "").strip().split(",")
    try:
        real_path = resolve_file_path(file_name)
        result = perform_correlation_analysis(real_path, target, features)
        output = []
        for feat, stats in result.items():
            output.append(
                f"{feat}:\n"
                f"  Pearson: {stats['pearson']} ({stats['pearson_interpretation']})\n"
                f"  Spearman: {stats['spearman']} ({stats['spearman_interpretation']})"
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error: {str(e)}"

# ===========================================================================================

# def predict_future(df, target_column, feature_columns):
#     df = df.dropna(subset=[target_column] + feature_columns)
#     X = df[feature_columns]
#     y = df[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     mse = mean_squared_error(y_test, preds)
#     return {"model": model, "mse": mse, "prediction_example": preds[:5]}



# def basic_eda(df):
#     return {
#         "head": df.head(),
#         "description": df.describe(),
#         "missing_values": df.isnull().sum(),
#         "correlation": df.corr(numeric_only=True)
#     }

# def generate_charts(df):
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     for col in numeric_cols:
#         plt.figure(figsize=(6, 4))
#         sns.histplot(df[col].dropna(), kde=True)
#         plt.title(f'Distribution of {col}')
#         plt.show()

# ===========================================================================================

class ANNClassifier:
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)
        self.model = None
        self.scaler = None

    def prepare_data(self, target_column: str, threshold: float = None):
        df = self.df.copy()

        # Drop non-numeric and irrelevant columns
        df = df.select_dtypes(include=[np.number])
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        if threshold is None:
            threshold = df[target_column].median()

        df['target_class'] = df[target_column].apply(lambda x: 1 if x >= threshold else 0)

        X = df.drop(columns=[target_column, 'target_class'])
        y = df['target_class']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42), threshold

    def train_ann(self, target_column: str, threshold: float = None):
        (X_train, X_test, y_train, y_test), actual_threshold = self.prepare_data(target_column, threshold)

        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Return a structured result
        return {
            "target": target_column,
            "accuracy": accuracy,
            "predictions": y_pred,
            "threshold": actual_threshold
        }

@tool
def get_ann_tool(file_name: str) -> Tool:
    """
    Returns a Tool that performs ANN-based classification on a specified numeric target column.
    Users must specify the target column using: 'Classify based on <column_name>'.
    """
    real_path = resolve_file_path(file_name)
    classifier = ANNClassifier(real_path)

    def _run(query: str) -> str:
        import re

        # Extract dynamic target column
        match = re.search(r"classify.*based on (.+)", query.lower())
        if not match:
            return "Please specify a target column in the format: 'Classify based on <column_name>'."

        target_column = match.group(1).strip()

        try:
            # Run the classification
            result = classifier.train_ann(target_column=target_column)

            return (
                f"✅ Classification complete for: {result['target']}\n"
                f"🔸 Accuracy: {result['accuracy']*100:.2f}%\n"
                f"📘 Threshold used: {result['threshold']}\n"
                f"🔮 Predictions: {result['predictions'][:10]}..."  # Display the first 10 predictions
            )
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return Tool(
        name="ANNClassifierTool",
        func=_run,
        description=(
            "Use this tool to run ANN classification on any numeric target column. "
            "Query format: 'Classify based on <column_name>'."
        )
    )


# ===========================================================================================

class DatasetReportGenerator:
    def __init__(self, data_path, model="deepseek-r1-distill-llama-70b"):
        self.df = pd.read_csv(data_path)
        self.llm = ChatGroq(temperature=0.3, model_name=model)

    def generate_report(self, _: str) -> str:
        df = self.df
        numeric = df.select_dtypes(include='number')
        summary = numeric.describe().T.round(2).to_string()
        columns = df.columns.tolist()

        prompt = PromptTemplate(
            input_variables=["columns", "summary"],
            template="""
You are a data analyst. Based on the dataset below, create a professional business report including:

1. 📈 Executive Summary
2. 🔢 Key Metrics
3. 📊 Trends or Observations
4. 🤝 Feature Relationships
5. 📌 Recommendations

Dataset Columns:
{columns}

Summary Statistics:
{summary}

Write the report clearly and concisely:
"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(columns=", ".join(columns), summary=summary)
    

@tool
def get_report_generator_tool(file_name: str) -> Tool:
    """Returns a Tool that generates a full structured report from the dataset, including summary, trends, and recommendations."""
    real_path = resolve_file_path(file_name)
    report_generator = DatasetReportGenerator(real_path)

    return Tool(
        name="LLMDatasetReportTool",
        func=report_generator.generate_report,
        description="Generates a full structured report (summary, trends, recommendations) from the dataset."
    )

# ===========================================================================================
    
class HistoricalQueryTool:
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def handle_query(self, query: str) -> str:
        df = self.df.copy()
        query = query.lower()
        columns = df.columns.tolist()

        # Extract year
        year_match = re.search(r"\b(20\d{2})\b", query)
        year = int(year_match.group()) if year_match else None

        # Match column using fuzzy match
        matched_column = None
        for word in query.split():
            close_matches = get_close_matches(word, columns, n=1, cutoff=0.6)
            if close_matches:
                matched_column = close_matches[0]
                break

        if not matched_column:
            return "❌ Couldn't detect a relevant column from your query."

        # Filter by year
        if year:
            if 'year' in df.columns:
                df = df[df['year'] == year]
            else:
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df[df[date_col].dt.year == year]

        # Handle different operations
        if "average" in query or "mean" in query:
            return f"📊 The average of **{matched_column}**{f' in {year}' if year else ''} is `{df[matched_column].mean():.2f}`"
        
        if "total" in query or "sum" in query:
            return f"🧮 The total of **{matched_column}**{f' in {year}' if year else ''} is `{df[matched_column].sum():,.2f}`"
        
        if "highest" in query or "maximum" in query or "max" in query:
            return f"🔺 The highest value of **{matched_column}**{f' in {year}' if year else ''} is `{df[matched_column].max():,.2f}`"
        
        if "lowest" in query or "minimum" in query or "min" in query:
            return f"🔻 The lowest value of **{matched_column}**{f' in {year}' if year else ''} is `{df[matched_column].min():,.2f}`"
        
        if "count" in query or "how many" in query:
            return f"🔢 Number of entries for **{matched_column}**{f' in {year}' if year else ''} is `{df[matched_column].count():,}`"

        return "❓ Sorry, I couldn't understand your historical question."
    
@tool
def get_history_tool(file_name: str) -> Tool:
    """
    Returns a Tool for answering historical data questions such as averages, totals, minimums, and maximums.
    Ideal for queries like 'What was the average market share in 2023?'
    """
    real_path = resolve_file_path(file_name)
    tool_instance = HistoricalQueryTool(real_path)

    return Tool(
        name="HistoricalInsightsTool",
        func=tool_instance.handle_query,
        description=(
            "Use this tool to ask historical questions about numeric values, such as averages, totals, max/min, or counts. "
            "Example: 'What was the average market share in 2023?'"
        )
    )

# ===========================================================================================

class SimpleLLMSummarizer:
    def __init__(self, file_path, model="deepseek-r1-distill-llama-70b"):
        self.df = pd.read_csv(file_path)
        self.llm = ChatGroq(temperature=0.3, model_name=model)

    def summarize_dataset(self, _: str) -> str:

        df = self.df
        columns = df.columns.tolist()
        #summary_stats = df.describe(include='all').to_string()

        prompt = PromptTemplate(
            input_variables=["columns", "stats"],
            template="""
You are a data expert. The user has uploaded a dataset.

Here are the column names:
{columns}


Please write a simple and clear summary of what this dataset contains, highlighting any interesting observations.
"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(columns=", ".join(columns))
    
@tool
def get_simple_summary_tool(file_name: str) -> Tool:
    """
    Returns a Tool that generates a basic LLM-powered summary of the uploaded dataset.
    Useful for quickly understanding the overall structure and content.
    """
    real_path = resolve_file_path(file_name)
    summarizer = SimpleLLMSummarizer(real_path)

    return Tool(
        name="LLMDatasetSummarizer",
        func=summarizer.summarize_dataset,
        description="Use this tool to get a basic LLM-generated summary of the uploaded dataset."
    )

# ===========================================================================================

class DynamicANNPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.model = None
        self.scaler = None

    def parse_query(self, query: str) -> str:
        keywords = [
            "predict", "classify", "forecast", "estimate", "determine"
        ]
        # Attempt to extract column name
        for col in self.df.columns:
            if any(k in query.lower() for k in keywords) and col.lower() in query.lower():
                return col
        raise ValueError("Could not extract target column from query. Please specify clearly.")

    def prepare_data(self, target_column: str, threshold: float = None):
        df = self.df.select_dtypes(include=[np.number]).copy()

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        if threshold is None:
            threshold = df[target_column].median()

        df["target_class"] = df[target_column].apply(lambda x: 1 if x >= threshold else 0)

        X = df.drop(columns=[target_column, "target_class"])
        y = df["target_class"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42), threshold

    def train_and_predict(self, query: str):
        target_column = self.parse_query(query)

        (X_train, X_test, y_train, y_test), threshold = self.prepare_data(target_column)

        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        high_pct = round(100 * sum(y_pred) / len(y_pred), 2)
        low_pct = round(100 - high_pct, 2)

        interpretation = (
            f"📌 Based on your data, the ANN model predicts:\n"
            f"- {high_pct}% of cases will have **HIGH** {target_column} (above threshold of {threshold})\n"
            f"- {low_pct}% will be **LOW**.\n\n"
            f"🧠 This means: most upcoming instances are likely to show "
            f"{'strong' if high_pct > 50 else 'weak'} performance for `{target_column}`."
        )

        return interpretation
   
@tool
def get_dynamic_prediction_tool(file_name: str) -> Tool:
    """
    Returns a Tool that uses an ANN model to make predictions from the dataset
    based on a user query (e.g., 'Predict high or low CLI this week').
    """
    real_path = resolve_file_path(file_name)
    predictor = DynamicANNPredictor(real_path)

    def _run(query: str) -> str:
        try:
            return predictor.train_and_predict(query)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return Tool(
        name="DynamicANNPredictionTool",
        func=_run,
        description="Use this tool to make predictions based on user query and dataset. Example: 'Predict high or low CLI this week'"
    )

# ===========================================================================================

# @tool
# def prediction_tool(input_str: str) -> str:
#     """Builds a prediction model for future estimates."""
#     params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
#     file_path = params.get("file_path", "").strip()
#     try:
#         real_path = resolve_file_path(file_path)
#         df = pd.read_csv(real_path)

#     except Exception as e:
#         return f"Error loading file: {str(e)}"
#     return handle_prediction_questions(params.get("query", ""), df)


# ============================== TOOL WRAPPERS ==============================

from django.core.files.storage import default_storage
from ..models import UploadedFile  # Import your Django model


# @tool
# def analyze_uploaded_file(input_str: str) -> str: 
#     """Analyze an uploaded file given its file_id."""
#     file_id = input_str.strip()  # The input_str is expected to be the file_id

#     if not file_id:
#         return "Please provide a file ID."

#     try:
#         uploaded_file_instance = UploadedFile.objects.get(id=file_id) 
#         file_path = default_storage.path(uploaded_file_instance.file.name)  
#         df = pd.read_csv(file_path)
#         return f"Analysis results:\n\n{df.head().to_markdown(index=False, numalign='left', stralign='left')}"
#     except UploadedFile.DoesNotExist:
#         return f"File not found with ID: {file_id}"
#     except Exception as e:
#         return f"Error analyzing file: {e}"




# def handle_prediction_questions(query: str, df: pd.DataFrame) -> str:
#     query_lower = query.lower()
#     columns = df.columns.tolist()
#     if not any(word in query_lower for word in ["predict", "forecast", "estimate", "future"]):
#         return "This doesn't look like a prediction query."
#     mentioned = [col for col in columns if col.lower() in query_lower]
#     if not mentioned:
#         return "Couldn't detect a target column."
#     target = mentioned[-1]
#     features = [col for col in columns if col != target]
#     try:
#         result = predict_future(df, target, features)
#         preds = ", ".join([f"{p:.2f}" for p in result["prediction_example"]])
#         return f"Model trained to predict `{target}`\n📊 MSE: {result['mse']:.2f}\nPredictions: {preds}"
#     except Exception as e:
#         return f"Error: {str(e)}"


#  ============================== NEW: ANN CLASSIFIER ==============================
