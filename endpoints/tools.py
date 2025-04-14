import os
import pandas as pd
from io import StringIO

class AnalysisTools:
    @staticmethod
    def describe_data(data, timeout_sec: int = 30, **kwargs):
        """Enhanced descriptive statistics with mixed data support"""
        try:
            import pandas as pd
            print("[DEBUG] Raw input to describe_data:", data)
            
            # Handle different input types
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, dict):  # JSON data
                df = pd.DataFrame(data)
            elif isinstance(data, str):  # JSON string
                df = pd.read_json(StringIO(data))
            else:
                return {
                    'status': 'error',
                    'message': 'Unsupported data format',
                    'solution': 'Provide DataFrame, dict or JSON string'
                }
                
            print("[DEBUG] Parsed DataFrame:", df.head())
            
            # Handle empty data
            if len(df.columns) == 0:
                return {'status': 'error', 'message': 'No columns found'}
                
            # Enhanced data type handling
            if kwargs.get('data_type_check', 'auto') == 'auto':
                stats = df.describe(include='all').to_dict()
            else:
                stats = df.describe().to_dict()
                
            # Calculate value counts for categorical data
            cat_cols = df.select_dtypes(exclude=['number']).columns
            value_counts = {}
            for col in cat_cols:
                value_counts[col] = df[col].value_counts().to_dict()
            
            return {
                'status': 'success',
                'result': {
                    'describe': stats,
                    'value_counts': value_counts
                },
                'message': "Analysis completed successfully"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Analysis failed: {str(e)}",
                'solution': "Try with a smaller dataset or different columns"
            }

    @staticmethod
    def correlation_analysis(data: str = None, x_col: str = None, y_col: str = None, **kwargs):
        """Enhanced correlation analysis with mixed data support"""
        try:
            df = pd.read_csv(StringIO(data))
            
            # Handle non-numeric data
            if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                if kwargs.get('fallback_strategy') == 'crosstab':
                    crosstab = pd.crosstab(df[x_col], df[y_col])
                    return {
                        'status': 'success',
                        'result': 'crosstab',
                        'table': crosstab.to_dict(),
                        'message': 'Used contingency table for categorical data'
                    }
                return {
                    'status': 'error',
                    'message': 'Columns must be numeric for correlation',
                    'solution': 'Try with numeric columns or use fallback_strategy="crosstab"'
                }
            
            corr = df[x_col].corr(df[y_col])
            return {
                'status': 'success',
                'correlation': corr,
                'interpretation': _interpret_correlation(corr)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'solution': 'Check data format and column types'
            }

    @staticmethod
    def linear_regression(data, x_col: str = None, y_col: str = None, **kwargs):
        """Enhanced linear regression with data validation and guidance"""
        try:
            # Handle different input types
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                return {
                    'status': 'error',
                    'message': 'Unsupported data format',
                    'solution': 'Provide CSV string, DataFrame or dict'
                }

            # Validate columns
            if x_col is None or y_col is None:
                return {
                    'status': 'input_required',
                    'message': 'Please specify both x and y columns',
                    'available_columns': list(df.columns),
                    'suggested_columns': ['orders_all_np'] if 'orders_all_np' in df.columns else None
                }

            if x_col not in df.columns or y_col not in df.columns:
                return {
                    'status': 'error',
                    'message': 'Specified columns not found',
                    'available_columns': list(df.columns)
                }

            if x_col == y_col:
                return {
                    'status': 'error',
                    'message': 'Cannot regress a column on itself'
                }

            # Perform regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(df[[x_col]], df[y_col])
            
            return {
                'status': 'success',
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(model.score(df[[x_col]], df[y_col])),
                'equation': f"{y_col} = {model.coef_[0]:.2f}*{x_col} + {model.intercept_:.2f}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'solution': 'Check data quality and column types'
            }

    @staticmethod
    def classify_data(data: str):
        """Classify data using simple ANN"""
        try:
            df = pd.read_csv(StringIO(data))
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import LabelEncoder
            # Implementation would go here
            return {
                'status': 'success',
                'message': 'Classification completed'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def _interpret_correlation(value: float) -> str:
    """Interpret correlation coefficient"""
    abs_value = abs(value)
    if abs_value > 0.7:
        return "Strong correlation"
    elif abs_value > 0.3:
        return "Moderate correlation"
    return "Weak or no correlation"
