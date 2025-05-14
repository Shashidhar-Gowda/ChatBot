

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
from typing import List  
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings
from typing import Union
from langchain.tools import tool
from fuzzywuzzy import process
from .column_matcher import find_best_column_match, match_columns # Import the column matcher


# Instantiate LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ============================== BASE FUNCTIONS ==============================

def resolve_file_path(file_name: str) -> str:
    """Resolve file path with Django media directory handling"""
    # Check in user_uploads directory first
    user_uploads_path = os.path.join(settings.MEDIA_ROOT, 'user_uploads', file_name)
    if os.path.exists(user_uploads_path):
        return user_uploads_path
    
    # Check in media root directly
    media_path = os.path.join(settings.MEDIA_ROOT, file_name)
    if os.path.exists(media_path):
        return media_path
    
    # Check current directory (for backward compatibility)
    current_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(current_path):
        return current_path
    
    # Return the most likely path for error reporting
    return user_uploads_path

def get_column_names(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return f"Columns: {', '.join(df.columns)}"



# def tool_error_handler(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except pd.errors.EmptyDataError:
#             return "Error: The file is empty or corrupt."
#         except FileNotFoundError:
#             return "Error: File not found. Please check the file path."
#         except Exception as e:
#             print(f"Tool error: {e}")
#             return f"Error processing request: {str(e)}"
#     return wrapper


# ===========================================================================================

def perform_linear_regression(file_path: str, target: str, features: List[str]) -> dict:
    df = pd.read_csv(file_path)
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

@tool()
def linear_regression_tool(input_str: str) -> str:
    """
    Perform linear regression on a given CSV file.
    
    Format: 'file_path=filename.csv; target=target; features=feat1,feat2,...'
    """
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = [f.strip() for f in params.get("features", "").split(",") if f.strip()]
    
    try:
        # Resolve file path
        real_path = resolve_file_path(file_name)
        df = pd.read_csv(real_path)
        actual_cols = df.columns.tolist()
        
        # Match target column
        matched_target = find_best_column_match(target, actual_cols)
        if not matched_target:
            return f"Target column '{target}' not found. Available columns: {actual_cols}"
        
        # Match feature columns (if applicable)
        if features:
            matched_features = []
            for feat in features:
                matched = find_best_column_match(feat, actual_cols)
                if not matched:
                    return f"Feature column '{feat}' not found. Available columns: {actual_cols}"
                matched_features.append(matched)
        

        # Run regression
        result = perform_linear_regression(real_path, matched_target, matched_features)
        return (
            f"Regression Analysis:\n"
            f"Target: {matched_target}\n"
            f"Features: {matched_features}\n"
            f"Equation: {result['equation']}\n"
            f"R² Score: {result['r2_score']:.4f}"
        )
        
    except Exception as e:
        return f"Error: {str(e)}"


# ===========================================================================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def perform_polynomial_regression(file_path: str, target: str, features: list, degree: int) -> str:
    try:
        df = pd.read_csv(file_path)

        # Validate columns
        if target not in df.columns:
            return f"Error: Target column '{target}' not found."
        for feat in features:
            if feat not in df.columns:
                return f"Error: Feature column '{feat}' not found."

        X = df[features].values
        y = df[target].values

        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        score = model.score(X_poly, y)
        coefs = model.coef_
        intercept = model.intercept_

        # Build equation string (assuming one feature)
        equation_terms = [f"{intercept:.2f}"]
        for i in range(1, len(coefs)):
            power = i
            coef = coefs[i]
            if coef == 0:
                continue
            if power == 1:
                term = f"{coef:+.4f}·x"
            else:
                term = f"{coef:+.6f}·x^{power}"
            equation_terms.append(term)
        equation = " ".join(equation_terms)

        # Interpret R²
        if score < 0.2:
            comment = "📉 The model does not fit the data well — very weak explanatory power."
        elif score < 0.5:
            comment = "📈 The model explains some variation, but may not be a strong fit."
        else:
            comment = "✅ The model fits the data reasonably well."

        return (
            f"🎯 Target: {target}\n📌 Feature: {features[0]}\n"
            f"📐 Polynomial Degree: {degree}\n\n"
            f"🧮 Model Equation:\n{target} = {equation}\n\n"
            f"📊 R² Score: {score:.3f}\n\n"
            f"{comment}\n\n"
            f"Would you like to try a different degree or add more features?"
        )

    except Exception as e:
        return f"Error during polynomial regression: {str(e)}"

@tool(return_direct=True)
def polynomial_regression_tool(input_str: str) -> str:
    """Perform polynomial regression. 
    Format: 'file_path=path; target=target; features=feat1,feat2,...; degree=int'
    If features are not specified, uses all numeric columns except target.
    """
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = [f.strip() for f in params.get("features", "").split(",") if f.strip()]
    degree = int(params.get("degree", "2").strip())
    
    try:
        # Resolve file path
        real_path = resolve_file_path(file_name)
        if not os.path.exists(real_path):
            available_files = [
                f for f in os.listdir(os.path.join(settings.MEDIA_ROOT, 'user_uploads')) 
                if f.endswith('.csv')
            ]
            return (
                f"File '{file_name}' not found in uploads directory.\n"
                f"Available files: {available_files or 'None found'}"
            )
        
        # Load data and get columns
        df = pd.read_csv(real_path)
        actual_cols = df.columns.tolist()
        
        # Match target column with better error handling
        matched_target = find_best_column_match(target, actual_cols)
        if not matched_target:
            return (
                f"Target column '{target}' not found. Available columns:\n"
                f"{', '.join(actual_cols)}"
            )
        
        # If no features specified, use all numeric columns except target
        if not features:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            features = [col for col in numeric_cols if col != matched_target]
            if not features:
                return "Error: No suitable numeric features found in the dataset."
        
        # Match feature columns with better feedback
        matched_features = []
        unmatched_features = []
        
        for feat in features:
            matched = find_best_column_match(feat, actual_cols)
            if matched:
                matched_features.append(matched)
            else:
                unmatched_features.append(feat)
        
        if unmatched_features:
            return (
                f"Could not match these feature columns: {', '.join(unmatched_features)}\n"
                f"Available columns: {', '.join(actual_cols)}"
            )
        
        # Validate degree
        if degree < 1 or degree > 5:
            return "Error: Degree must be between 1 and 5 (inclusive)."
        
        # Run polynomial regression with matched columns
        return perform_polynomial_regression(
            real_path, 
            matched_target, 
            matched_features, 
            degree
        )
        
    except ValueError as e:
        return f"Input error: {str(e)}"
    except pd.errors.EmptyDataError:
        return "Error: The file is empty or corrupt."
    except Exception as e:
        return f"Error performing polynomial regression: {str(e)}"
    
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

@tool(return_direct=True)
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

@tool()
def correlation_tool(input_str: str) -> str:
    """Perform correlation analysis. Format: 'file_path=path; target=target; features=feat1,feat2,...'"""
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = [f.strip() for f in params.get("features", "").split(",") if f.strip()]
    
    try:
        # Resolve file path
        real_path = resolve_file_path(file_name)
        if not os.path.exists(real_path):
            available_files = [
                f for f in os.listdir(os.path.join(settings.MEDIA_ROOT, 'user_uploads')) 
                if f.endswith('.csv')
            ]
            return (
                f"File '{file_name}' not found in uploads directory.\n"
                f"Available files: {available_files or 'None found'}"
            )
        
        # Load data and get columns
        df = pd.read_csv(real_path)
        actual_cols = df.columns.tolist()
        
        # Match target column with improved error message
        matched_target = find_best_column_match(target, actual_cols)
        if not matched_target:
            return (
                f"Target column '{target}' not found. Available columns:\n"
                f"{', '.join(actual_cols)}"
            )
        
        # Match feature columns with better feedback
        matched_features = []
        unmatched_features = []
        
        for feat in features:
            matched = find_best_column_match(feat, actual_cols)
            if matched:
                matched_features.append(matched)
            else:
                unmatched_features.append(feat)
        
        if unmatched_features:
            return (
                f"Could not match these feature columns: {', '.join(unmatched_features)}\n"
                f"Available columns: {', '.join(actual_cols)}"
            )
        
        if not matched_features:
            return "Error: Please specify at least one feature column."
        
        # Run correlation analysis with matched columns
        result = perform_correlation_analysis(real_path, matched_target, matched_features)
        
        # Format output
        output = []
        for feat, stats in result.items():
            output.append(
                f"📊 {feat}:\n"
                f"  • Pearson: {stats['pearson']} ({stats['pearson_interpretation']})\n"
                f"  • Spearman: {stats['spearman']} ({stats['spearman_interpretation']})"
            )
        
        return (
            f"Correlation Analysis:\n"
            f"Target: {matched_target}\n"
            f"Features: {', '.join(matched_features)}\n\n" +
            "\n\n".join(output)
        )
        
    except pd.errors.EmptyDataError:
        return "Error: The file is empty or corrupt."
    except Exception as e:
        return f"Error performing correlation analysis: {str(e)}"
# ===========================================================================================

# ===========================================================================================

class ANNClassifier:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
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

        accuracy = accuracy_score(y_test, y_pred)

        return {
            "target": target_column,
            "accuracy": accuracy,
            "predictions": y_pred,
            "threshold": actual_threshold
        }


@tool(return_direct=True)
def get_ann_tool(input_str: str) -> str:
    """
    Perform ANN-based classification. Format: 'file_path=filename.csv; target=column_name'
    """
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target_column = params.get("target", "").strip()

    if not file_name or not target_column:
        return "Missing required parameters. Format: 'file_path=filename.csv; target=column_name'"

    try:
        # Resolve file path with better error handling
        real_path = resolve_file_path(file_name)
        if not os.path.exists(real_path):
            # Check available files in upload directory
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'user_uploads')
            available_files = []
            if os.path.exists(upload_dir):
                available_files = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
            
            return (
                f"File '{file_name}' not found in uploads directory.\n"
                f"Available CSV files: {', '.join(available_files) if available_files else 'None found'}"
            )

        # Load data and get columns
        df = pd.read_csv(real_path)
        actual_cols = df.columns.tolist()

        # Match target column with fuzzy matching
        matched_target = find_best_column_match(target_column, actual_cols)
        if not matched_target:
            return (
                f"Target column '{target_column}' not found. Available columns:\n"
                f"{', '.join(actual_cols)}"
            )

        # Validate target column is numeric
        if not pd.api.types.is_numeric_dtype(df[matched_target]):
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            return (
                f"Error: Target column '{matched_target}' must be numeric for ANN classification.\n"
                f"Available numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None found'}"
            )

        # Run ANN classification with matched target column
        classifier = ANNClassifier(real_path)
        result = classifier.train_ann(target_column=matched_target)

        # Format output
        return (
            f"🧠 ANN Classification Results:\n\n"
            f"• Target Column: {matched_target}\n"
            f"• Model Accuracy: {result['accuracy']*100:.2f}%\n"
            f"• Classification Threshold (median): {result['threshold']:.4f}\n"
            f"• Class Distribution:\n"
            f"  - High Class (≥ threshold): {sum(result['predictions'])} cases\n"
            f"  - Low Class (< threshold): {len(result['predictions']) - sum(result['predictions'])} cases\n"
            f"• First 10 Predictions: {result['predictions'][:10].tolist()}"
        )

    except pd.errors.EmptyDataError:
        return "Error: The file is empty or contains no data."
    except ValueError as e:
        return f"Input error: {str(e)}"
    except Exception as e:
        return f"Error performing ANN classification: {str(e)}"




# ===========================================================================================

class DatasetReportGenerator:
    def __init__(self, file_path, model="deepseek-r1-distill-llama-70b"):
        self.df = pd.read_csv(file_path)
        self.llm = ChatGroq(temperature=0.3, model_name=model)

    def generate_report(self, _: str) -> str:
        try:
            df = self.df
            numeric = df.select_dtypes(include='number')
            summary = numeric.describe().T.round(2).to_string()
            columns = df.columns.tolist()

            prompt = PromptTemplate(
                input_variables=["columns", "summary"],
                template="""
You are a data analyst. Based on the dataset below, create a professional business report including:

1. Executive Summary
2. Key Metrics 
3. Trends or Observations
4. Feature Relationships
5. Recommendations

Dataset Columns:
{columns}

Summary Statistics:
{summary}

Write the report clearly and concisely:
"""
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(columns=", ".join(columns), summary=summary)

        except Exception as e:
            return f"Report generation failed: {str(e)}"

@tool(return_direct=True)
def get_report_generator_tool(input_str: str) -> str:
    """Generate a structured report from a dataset. 
    Input must be in one of these formats:
    - 'file_path=filename.csv'
    - 'filename.csv'
    
    Example: 
    - 'file_path=your_data.csv' 
    - 'sales_data.csv'"""
    
    try:
        # Handle empty input case first
        if not input_str.strip():
            return (
                "⚠️ Please provide a filename.\n"
                "Valid formats:\n"
                "- 'file_path=data.csv'\n"
                "- 'data.csv'\n\n"
                "Example: 'file_path=your_assignment_data.csv'"
            )

        # Parse input with better error handling
        if 'file_path=' in input_str:
            try:
                params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
                file_name = params.get("file_path", "").strip()
            except Exception:
                file_name = input_str.replace('file_path=', '').strip()
        else:
            file_name = input_str.strip()

        if not file_name:
            return (
                "Invalid input format.\n"
                "Please use one of these formats:\n"
                "- 'file_path=filename.csv'\n"
                "- 'filename.csv'\n\n"
                "Example: 'sales_data.csv'"
            )

        # File handling with comprehensive checks
        real_path = resolve_file_path(file_name)
        if not os.path.exists(real_path):
            # Get available files with proper error handling
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'user_uploads')
            try:
                available_files = [
                    f for f in os.listdir(upload_dir) 
                    if f.endswith('.csv') and not f.startswith('.')
                ] if os.path.exists(upload_dir) else []
            except Exception as e:
                available_files = []
                print(f"Error reading upload directory: {str(e)}")

            return (
                f"🔍 File '{file_name}' not found.\n\n"
                f"Available CSV files:\n"
                f"{chr(10).join(f'• {f}' for f in available_files) if available_files else '• No CSV files available'}\n\n"
                f"Please upload your file first or check the filename."
            )

        # Validate file content
        if os.path.getsize(real_path) == 0:
            return "Error: The file is empty. Please upload a valid CSV file with data."

        try:
            # Quick check if file is readable CSV
            pd.read_csv(real_path, nrows=1)
        except pd.errors.ParserError:
            return "Error: Invalid CSV format. Please check the file."
        except UnicodeDecodeError:
            return "Error: File encoding issue. Please save as UTF-8 CSV."

        # Generate report with matched file
        report_generator = DatasetReportGenerator(real_path)
        report = report_generator.generate_report("")
        
        return (
            f"📊 Dataset Analysis Report\n"
            f"File: {file_name}\n\n"
            f"{report}\n\n"
            f"💡 Need more specific insights? You can ask about:\n"
            f"- Correlations between columns\n"
            f"- Statistical summaries\n"
            f"- Data visualizations"
        )

    except pd.errors.EmptyDataError:
        return "Error: The file contains no data or has invalid format."
    except Exception as e:
        return (
            f"Report generation failed.\n\n"
            f"Possible issues:\n"
            f"1. File is not a valid CSV\n"
            f"2. File contains corrupted data\n"
            f"3. File permissions issue\n\n"
            f"Technical details: {str(e)}"
        )
# ===========================================================================================
from difflib import get_close_matches
import pandas as pd
import re

class HistoricalQueryTool:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.columns = self.df.columns.tolist()

    def handle_query(self, query: str) -> str:
        df = self.df.copy()
        query = query.lower()

        # Extract the year from the query if present
        year_match = re.search(r"\b(20\d{2})\b", query)
        year = int(year_match.group()) if year_match else None

        # Find the most relevant column using fuzzy matching
        possible_revenue_keywords = ["revenue", "sales", "spend", "income", "earnings", "profit"]
        matched_column = None
        for word in query.split():
            # Use fuzzy matching for each keyword
            for keyword in possible_revenue_keywords:
                close_matches = get_close_matches(keyword, self.columns, n=1, cutoff=0.4)
                if close_matches:
                    matched_column = close_matches[0]
                    break
            if matched_column:
                break

        if not matched_column:
            return "Couldn't detect a relevant revenue-related column from your query."

        # Filter by year if needed
        if year and 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
            df = df[df['report_date'].dt.year == year]

        # Calculate the total revenue
        if matched_column in df.columns:
            total_revenue = df[matched_column].sum()
            return (
                f"📊 Historical Data Response:\n\n"
                f"• Matched Column: {matched_column}\n"
                f"• Total {matched_column} in **{year if year else 'all years'}**: `{total_revenue:,.2f}`\n"
                f"• Data Type: {df[matched_column].dtype}\n"
                f"• Sample Data: {df[matched_column].head(5).tolist()}"
            )
        else:
            return f"Column **{matched_column}** not found in the data."

        return "Sorry, I couldn't understand your historical question."



    
@tool(return_direct=True)
def get_history_tool(input_str: str) -> str:
    """
    Answer historical questions. Format: 'file_path=filename.csv; question=What was the total revenue in 2023?'
    """
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    query = params.get("question", "").strip()

    if not file_name or not query:
        return "Please provide both file_path and question in the format: 'file_path=...; question=...'."

    try:
        # Resolve file path
        file_path = resolve_file_path(file_name)

        # Load data
        df = pd.read_csv(file_path)
        actual_cols = df.columns.tolist()

        # Attempt to match query keywords with columns
        matched_column = find_best_column_match(query, actual_cols)
        if not matched_column:
            return (
                f"No matching column found for the question '{query}'.\n"
                f"Available columns: {', '.join(actual_cols)}"
            )

        # Generate response based on matched column
        return (
            f"📊 Historical Data Response:\n\n"
            f"• Matched Column: {matched_column}\n"
            f"• Total {matched_column} in the dataset: {df[matched_column].sum()}\n"
            f"• Data Type: {df[matched_column].dtype}\n"
            f"• Sample Data: {df[matched_column].head(5).tolist()}"
        )

    except FileNotFoundError as e:
        # Handle missing files
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'user_uploads')
        available_files = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
        return (
            f"File '{file_name}' not found in the uploads directory.\n"
            f"Available CSV files: {', '.join(available_files) if available_files else 'None found'}"
        )
    except pd.errors.EmptyDataError:
        return "Error: The file is empty or contains no data."
    except Exception as e:
        return f"Error processing historical data: {str(e)}"



# ===========================================================================================

class SimpleLLMSummarizer:
    def __init__(self, file_path, model="deepseek-r1-distill-llama-70b"):
        self.df = pd.read_csv(file_path)
        self.llm = ChatGroq(temperature=0.3, model_name=model)

    def summarize_dataset(self, _: str) -> str:
        df = self.df
        columns = df.columns.tolist()

        prompt = PromptTemplate(
            input_variables=["columns"],
            template="""
You are a data expert. The user has uploaded a dataset.

Here are the column names:
{columns}

Please write a simple and clear summary of what this dataset contains, highlighting any interesting observations.
"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            summary = chain.run(columns=", ".join(columns))
            return summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"

    
@tool(return_direct=True)
def get_simple_summary_tool(input_str: str) -> str:
    """
    Generate a simple LLM-based summary of a dataset.
    Format: 'file_path=filename.csv; columns=col1,col2,...'
    """
    import re
    import os
    import pandas as pd
    
    # Extract parameters from input string
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    column_names = [col.strip() for col in params.get("columns", "").split(",") if col.strip()]
    
    if not file_name:
        return "Please provide file_path in the format: 'file_path=filename.csv'"
    
    try:
        # Resolve the file path
        file_path = resolve_file_path(file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            available_files = [
                f for f in os.listdir(os.path.join(settings.MEDIA_ROOT, 'user_uploads')) 
                if f.endswith('.csv')
            ]
            return (
                f"File '{file_name}' not found in uploads directory.\n"
                f"Available files: {available_files or 'None found'}"
            )
        
        # Load the dataset
        df = pd.read_csv(file_path)
        actual_cols = df.columns.tolist()
        
        # Match requested columns
        matched_columns = []
        unmatched_columns = []
        
        for col in column_names:
            matched = find_best_column_match(col, actual_cols)
            if matched:
                matched_columns.append(matched)
            else:
                unmatched_columns.append(col)
        
        if unmatched_columns:
            return (
                f"Could not match these columns: {', '.join(unmatched_columns)}\n"
                f"Available columns: {', '.join(actual_cols)}"
            )
        
        # Summarize the matched columns
        summarizer = SimpleLLMSummarizer(file_path)
        summary = summarizer.summarize_dataset(", ".join(matched_columns))
        return summary
        
    except pd.errors.EmptyDataError:
        return "Error: The file is empty or corrupt."
    except Exception as e:
        return f"Error: {str(e)}"




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
            f"Based on your data, the ANN model predicts:\n"
            f"- {high_pct}% of cases will have **HIGH** {target_column} (above threshold of {threshold})\n"
            f"- {low_pct}% will be **LOW**.\n\n"
            f"This means: most upcoming instances are likely to show "
            f"{'strong' if high_pct > 50 else 'weak'} performance for `{target_column}`."
        )

        return interpretation


@tool(return_direct=True)
def get_dynamic_prediction_tool(input_str: str) -> str:
    """
    Uses an ANN to classify a target column into high/low classes based on a query.
    Format: 'file_path=filename.csv; target=target_column; features=feat1,feat2,...; question=your prediction query'
    Example: 'file_path=data.csv; target=Revenue; features=Year,Month,Sales; question=Predict sales for next quarter'
    """
    import re
    import os
    import pandas as pd
    
    # Extract parameters from input string
    params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
    file_name = params.get("file_path", "").strip()
    target = params.get("target", "").strip()
    features = [f.strip() for f in params.get("features", "").split(",") if f.strip()]
    query = params.get("question", "").strip()

    # Check required parameters
    if not file_name or not target or not features or not query:
        return (
            "Please provide all required parameters in the format:\n"
            "'file_path=filename.csv; target=target_column; features=feat1,feat2,...; question=your query'"
        )
    
    try:
        # Resolve the file path
        file_path = resolve_file_path(file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            available_files = [
                f for f in os.listdir(os.path.join(settings.MEDIA_ROOT, 'user_uploads')) 
                if f.endswith('.csv')
            ]
            return (
                f"File '{file_name}' not found in uploads directory.\n"
                f"Available files: {available_files or 'None found'}"
            )
        
        # Load the dataset
        df = pd.read_csv(file_path)
        actual_cols = df.columns.tolist()
        
        # Match the target column
        matched_target = find_best_column_match(target, actual_cols)
        if not matched_target:
            return (
                f"Target column '{target}' not found. Available columns:\n"
                f"{', '.join(actual_cols)}"
            )
        
        # Match the feature columns
        matched_features = []
        unmatched_features = []
        
        for feat in features:
            matched = find_best_column_match(feat, actual_cols)
            if matched:
                matched_features.append(matched)
            else:
                unmatched_features.append(feat)
        
        if unmatched_features:
            return (
                f"Could not match these feature columns: {', '.join(unmatched_features)}\n"
                f"Available columns: {', '.join(actual_cols)}"
            )
        
        if not matched_features:
            return "Error: Please specify at least one feature column."
        
        # Train and predict using the matched columns
        predictor = DynamicANNPredictor(file_path)
        prediction = predictor.train_and_predict(
            query=query,
            target_column=matched_target,
            feature_columns=matched_features
        )
        
        return prediction
        
    except pd.errors.EmptyDataError:
        return "Error: The file is empty or corrupt."
    except Exception as e:
        return f"Error: {str(e)}"



# # ======================================================================
# llm_visualizer.py
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence 



class LLMVisualizationGenerator:
    def __init__(self, file_path, model="deepseek-r1-distill-llama-70b"):
        self.df = pd.read_csv(file_path)
        self.llm = ChatGroq(temperature=0.3, model_name=model)

    def generate_code(self, question: str) -> str:
        columns = self.df.columns.tolist()
        prompt = PromptTemplate(
    input_variables=["columns", "question"],
    template="""
You are a Python data visualization expert. The dataset has the following columns:
{columns}

User's request: {question}

Your job:
- Return only a single valid Python code block (no explanations, no comments).
- The dataframe is named `df`.
- Use matplotlib or seaborn to create the plot.
- Import required libraries at the top.
- End with: plt.savefig("some_path.png")
- DO NOT include any explanations, comments, markdown, or text outside the code.

Example output format:
import matplotlib.pyplot as plt
import seaborn as sns
# (rest of code)
plt.savefig("some_path.png")
"""
)

        chain = prompt | self.llm
        result = chain.invoke({"columns": ", ".join(columns), "question": question})
        return result


import re

def extract_code_only(llm_output: str) -> str:
    # Extract code block between triple backticks, or fallback to plain
    matches = re.findall(r"```(?:python)?(.*?)```", llm_output, re.DOTALL)
    return matches[0].strip() if matches else llm_output.strip()


# In tools.py, update the get_visualization_tool function:

@tool(return_direct=True)
def get_visualization_tool(input_str: str) -> str:
    """This tool generates a visualization using LLM-generated code from a natural language prompt."""
    import os
    import re
    import uuid
    import json
    import pandas as pd
    from django.conf import settings
    from .utils import resolve_file_path
    from difflib import get_close_matches
    import glob
    import traceback

    try:
        # 1. Extract filename from prompt
        match = re.search(r'\bfrom\s+([a-zA-Z0-9_ -]+)', input_str.lower())
        file_hint = match.group(1).strip() if match else ""

        # 2. Get all available CSV files in media/user_uploads
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'user_uploads')
        all_csv_files = glob.glob(os.path.join(upload_dir, '*.csv'))
        all_csv_names = [os.path.basename(f) for f in all_csv_files]

        # 3. Try fuzzy match against filenames
        matched = get_close_matches(file_hint + ".csv", all_csv_names, n=1, cutoff=0.6)
        file_name = matched[0] if matched else ""

        if not file_name:
            return json.dumps({"error": f"Could not infer dataset from query: '{file_hint}'"})

        file_path = resolve_file_path(file_name)

        if not os.path.exists(file_path):
            return json.dumps({"error": f"Resolved file '{file_name}' not found at: {file_path}"})

        # 4. Generate LLM code for visualization
        generator = LLMVisualizationGenerator(file_path)
        code = generator.generate_code(input_str)

        # Ensure 'code' is a string
        if hasattr(code, 'content') and isinstance(code.content, str):
            code = code.content
        elif not isinstance(code, str):
            return json.dumps({"error": f"LLM did not return valid code. Got: {type(code)}", "code": str(code)})
        code = extract_code_only(code)
        if not isinstance(code, str):
            return json.dumps({"error": f"Code extraction failed. Got: {type(code)}", "code": code})

        # 5. Create plot path
        plot_filename = f"plot_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(settings.MEDIA_ROOT, 'media/plots', plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # 6. Inject the correct file path into the code
        replacement_string = file_path.replace("'", "\\'")
        code = code.replace(f"'{file_name}'", f"'{replacement_string}'")

        replacement_string = file_path.replace('"', '\\"')
        code = code.replace(f'"{file_name}"', f'"{replacement_string}"')

        # 7. Execute code
        local_env = {}  # Start with an EMPTY local environment
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': None,
            'plt': None,
            'sns': None,
            'df': None
        }

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            exec_globals['pd'] = pd
            exec_globals['plt'] = plt
            exec_globals['sns'] = sns

            # Load the DataFrame INSIDE the exec context
            exec_globals['df'] = pd.read_csv(file_path)

            print("\n--- Code about to be executed: ---")
            print(code)
            print("--- End of code ---")

            exec(code, exec_globals, local_env)

            print("\n--- Code executed successfully ---")

            # **CRITICAL: Check if 'df' and 'plt' are still valid AFTER exec**
            if exec_globals['df'] is None:
                raise ValueError("DataFrame 'df' became None during code execution.")
            if exec_globals['plt'] is None:
                raise ValueError("Matplotlib 'plt' became None during code execution.")

            # **CRITICAL: Check if the file was actually saved**
            if not os.path.exists(plot_path):
                raise RuntimeError(f"Plot file was not saved at: {plot_path}")

            print(f"Plot saved successfully at: {plot_path}")

        except Exception as e:
            error_message = f"Error executing generated code: {str(e)}"
            print(error_message)
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return json.dumps({"error": error_message, "code": code, "trace": traceback_str})

        # 8. Return result
        return json.dumps({
            "image_url": f"/media/plots/{plot_filename}",
            "type": "custom",
            "code_used": code
        })

    except Exception as e:
        return json.dumps({"error": f"Error during visualization generation: {str(e)}", "trace": traceback.format_exc()})

# # ======================================================================
# class VisualizationTool:
#     def __init__(self, data_path):
#         self.df = pd.read_csv(data_path)

#     def parse_columns(self, query: str):
#         cols = self.df.columns.tolist()
#         matched = []
#         for word in query.lower().split():
#             match = get_close_matches(word, cols, n=1, cutoff=0.6)
#             if match:
#                 matched.append(match[0])
#         return list(dict.fromkeys(matched))  # remove duplicates

#     def generate_plot_code(self, query: str):
#         columns = self.parse_columns(query)

#         if len(columns) == 0:
#             return "No matching columns found in your query."

#         if "histogram" in query.lower() or "distribution" in query.lower():
#             col = columns[0]
#             return f"""```python
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.hist(df['{col}'].dropna(), bins=30, color='skyblue', edgecolor='black')
# plt.title('Histogram of {col}')
# plt.xlabel('{col}')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# ```"""

#         if "scatter" in query.lower() or "plot" in query.lower() or len(columns) == 2:
#             if len(columns) < 2:
#                 return "Need two columns for a scatterplot."
#             col1, col2 = columns[:2]
#             return f"""```python
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.scatter(df['{col1}'], df['{col2}'], alpha=0.7, color='green')
# plt.title('Scatterplot of {col1} vs {col2}')
# plt.xlabel('{col1}')
# plt.ylabel('{col2}')
# plt.grid(True)
# plt.show()
# ```"""

#         return "Couldn't determine plot type. Please ask for 'histogram' or 'scatterplot'."

# @tool
# def get_visualization_tool(input_str: str, request) -> str:
#     """
#     Returns matplotlib code for histograms or scatterplots from your dataset.
#     Format: 'file_path=filename.csv; question=your visualization request'
#     Example: 'file_path=data.csv; question=Plot scatter between trust and share'
#     """
#     import re
#     params = dict(re.findall(r'(\w+)=([^;]+)', input_str))
#     file_name = params.get("file_path", "").strip()
#     query = params.get("question", "").strip()

#     if not file_name or not query:
#         return "Please provide both file_path and question in the format: 'file_path=filename.csv; question=your query'"

#     try:
#         file_path = resolve_file_path(file_name)
#         viz_tool = VisualizationTool(file_path)
#         return viz_tool.generate_plot_code(query)
#     except Exception as e:
#         return f"Error: {str(e)}"



