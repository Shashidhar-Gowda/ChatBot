# column_matcher.py
from fuzzywuzzy import process
import re
from typing import Tuple, List

def clean_column_name(col: str) -> str:
    """Keep spaces but normalize otherwise"""
    return re.sub(r'[^\w\s]', '', col.lower()).strip()

def correct_and_replace_columns(user_query: str, actual_cols: List[str], threshold: int = 80) -> Tuple[str, List[str]]:
    """
    Fuzzy-matches potential column references in user query to actual column names,
    and returns a cleaned query + list of matched column names.
    
    Args:
        user_query: The user's input query
        actual_cols: List of actual column names from the dataset
        threshold: Minimum fuzzy match score (0-100)
        
    Returns:
        Tuple of (cleaned_query, matched_columns)
    """
    cleaned_actual = [clean_column_name(c) for c in actual_cols]
    matched = {}
    
    # Find all potential column references in query
    words = re.findall(r'\b\w+\b', user_query)
    
    for word in words:
        cleaned_word = clean_column_name(word)
        # Skip very short words
        if len(cleaned_word) < 3:
            continue
            
        match, score = process.extractOne(cleaned_word, cleaned_actual)
        if score >= threshold:
            original_col = actual_cols[cleaned_actual.index(match)]
            matched[word] = original_col
    
    # Replace matches in query while preserving original case
    cleaned_query = user_query
    for original, corrected in matched.items():
        cleaned_query = re.sub(
            rf'(?<!\w){re.escape(original)}(?!\w)',  # Whole word match
            corrected,
            cleaned_query,
            flags=re.IGNORECASE
        )
    
    return cleaned_query, list(matched.values())