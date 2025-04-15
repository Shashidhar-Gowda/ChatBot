from langchain.tools import tool
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

@tool(description="Preprocesses the data by removing nulls and trimming string columns.")
def get_preprocessing_tool():
    def preprocess_data(df):
        df = df.dropna()
        df.columns = [col.strip() for col in df.columns]
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df
    return preprocess_data

@tool(description="Performs Pearson and Spearman correlation analysis on numeric columns.")
def get_correlation_tool():
    def correlation_analysis(df):
        numeric_df = df.select_dtypes(include=[np.number])
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')
        return f"Pearson Correlation:\n{pearson_corr}\n\nSpearman Correlation:\n{spearman_corr}"
    return correlation_analysis

@tool(description="Runs a linear regression to predict a target column using numeric features.")
def get_regression_tool():
    def linear_regression(df, target_column):
        df = df.select_dtypes(include=[np.number])
        if target_column not in df.columns:
            return f"Target column '{target_column}' not found in numeric columns."
        X = df.drop(columns=[target_column])
        y = df[target_column]
        model = LinearRegression()
        model.fit(X, y)
        return f"Coefficients: {dict(zip(X.columns, model.coef_))}\nIntercept: {model.intercept_}\nR² Score: {model.score(X, y)}"
    return linear_regression

@tool(description="Creates a pairplot of numeric variables and returns a base64-encoded PNG.")
def get_visualization_tool():
    def visualize_data(df):
        numeric_df = df.select_dtypes(include=[np.number])
        sns.set(style="ticks")
        plot = sns.pairplot(numeric_df)
        buf = BytesIO()
        plot.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return f"data:image/png;base64,{img_base64}"
    return visualize_data
