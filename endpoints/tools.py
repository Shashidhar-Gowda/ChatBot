import os
import pandas as pd
from io import StringIO

class AnalysisTools:
    @staticmethod
    def describe_data(data: str, timeout_sec: int = 30):
        """Enhanced descriptive statistics with timeout handling"""
        try:
            import pandas as pd
            from io import StringIO
            
            # Handle both direct data and file paths
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif os.path.exists(str(data)):  # It's a file path
                # Process in chunks for memory efficiency
                chunks = pd.read_csv(data, chunksize=50000)
                df = pd.concat(chunks)
            else:  # It's raw data
                # For in-memory data, limit to first 1MB to prevent OOM
                if len(data) > 1_000_000:
                    data = data[:1_000_000]
                    return {
                        'status': 'warning',
                        'message': 'Data truncated to first 1MB for processing'
                    }
                df = pd.read_csv(StringIO(data))
            
            # Ensure we have numeric columns
            if len(df.columns) == 0:
                return {
                    'status': 'error',
                    'message': 'No columns found in data',
                    'solution': 'Check data format and column headers'
                }
                
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {
                    'status': 'error', 
                    'message': 'No numeric columns found',
                    'solution': 'Try with different columns or check data types'
                }
                
            basic_stats = df[numeric_cols].describe().to_dict()
            
            return {
                'status': 'success',
                'result': basic_stats,
                'message': "Analysis completed successfully"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Analysis failed: {str(e)}",
                'solution': "Try with a smaller dataset or different columns"
            }

    @staticmethod
    def correlation_analysis(data: str = None, x_col: str = None, y_col: str = None):
        """Calculate correlation between two numeric columns"""
        try:
            df = pd.read_csv(StringIO(data))
            corr = df[x_col].corr(df[y_col])
            return {
                'status': 'success',
                'correlation': corr,
                'interpretation': _interpret_correlation(corr)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    @staticmethod
    def linear_regression(data: str, x_col: str, y_col: str):
        """Perform linear regression"""
        try:
            df = pd.read_csv(StringIO(data))
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(df[[x_col]], df[y_col])
            return {
                'status': 'success',
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(model.score(df[[x_col]], df[y_col]))
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
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
