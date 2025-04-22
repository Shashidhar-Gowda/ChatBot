import os
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

class AnalysisTools:
    @staticmethod
    def describe_data(data, timeout_sec: int = 30, **kwargs):
        """Enhanced descriptive statistics with timeout support"""
        try:
            if isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return "Unsupported data format for description."
            desc = df.describe(include='all').to_string()
            return desc
        except Exception as e:
            return f"Error in describe_data: {str(e)}"

    @staticmethod
    def correlation_analysis(data, x_col=None, y_col=None, **kwargs):
        """Perform correlation analysis between two columns"""
        try:
            if isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return "Unsupported data format for correlation."
            if x_col is None or y_col is None:
                return "Both x_col and y_col must be specified."
            corr = df[[x_col, y_col]].corr().to_string()
            return corr
        except Exception as e:
            return f"Error in correlation_analysis: {str(e)}"

    @staticmethod
    def linear_regression(data, target_col, feature_cols, **kwargs):
        """Perform linear regression on the data"""
        try:
            if isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return "Unsupported data format for linear regression."
            X = df[feature_cols]
            y = df[target_col]
            model = LinearRegression()
            model.fit(X, y)
            coef_str = ", ".join([f"{col}: {coef:.4f}" for col, coef in zip(feature_cols, model.coef_)])
            intercept = model.intercept_
            score = model.score(X, y)
            result = f"Intercept: {intercept:.4f}\nCoefficients: {coef_str}\nR^2 Score: {score:.4f}"
            return result
        except Exception as e:
            return f"Error in linear_regression: {str(e)}"

    @staticmethod
    def classify_data(data, target_col, feature_cols, **kwargs):
        """Perform simple classification using MLPClassifier"""
        try:
            if isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return "Unsupported data format for classification."
            X = df[feature_cols]
            y = df[target_col]
            clf = MLPClassifier(max_iter=500)
            clf.fit(X, y)
            score = clf.score(X, y)
            result = f"Classification accuracy: {score:.4f}"
            return result
        except Exception as e:
            return f"Error in classify_data: {str(e)}"
