# column_matcher.py
from fuzzywuzzy import process
import re
from typing import Tuple, List, Dict, Optional

def clean_column_name(col: str) -> str:
    """Normalize column names for matching"""
    return re.sub(r'[^\w\s]', '', col.lower()).strip()

def find_best_column_match(
    input_col: str, 
    available_cols: List[str],
    threshold: int = 80
) -> Optional[str]:
    """
    Find the best matching column name using fuzzy matching.
    
    Args:
        input_col: The column name to match
        available_cols: List of actual column names
        threshold: Minimum match score (0-100)
        
    Returns:
        The best matching column name or None if no good match found
    """
    input_clean = clean_column_name(input_col)
    available_clean = [(c, clean_column_name(c)) for c in available_cols]
    
    # Try exact match first
    for orig, clean in available_clean:
        if input_clean == clean:
            return orig
    
    # Try contains match
    for orig, clean in available_clean:
        if input_clean in clean:
            return orig
    
    # Try fuzzy match
    match, score = process.extractOne(
        input_clean, 
        [clean for _, clean in available_clean]
    )
    if score >= threshold:
        return available_cols[[clean for _, clean in available_clean].index(match)]
    return None

def match_columns(
    input_columns: List[str], 
    available_cols: List[str],
    threshold: int = 80
) -> Tuple[Dict[str, str], List[str]]:
    """
    Match a list of input columns to available columns.
    
    Args:
        input_columns: List of column names to match
        available_cols: List of actual column names
        threshold: Minimum fuzzy match score
        
    Returns:
        Tuple of (matched_columns, missing_columns)
        matched_columns: Dict of {input: matched_column}
        missing_columns: List of input columns with no good match
    """
    matched = {}
    missing = []
    
    for col in input_columns:
        matched_col = find_best_column_match(col, available_cols, threshold)
        if matched_col:
            matched[col] = matched_col
        else:
            missing.append(col)
            
    return matched, missing