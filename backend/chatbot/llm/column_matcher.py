from fuzzywuzzy import process
import re

def clean_column_name(col: str) -> str:
    return " ".join(col.lower().split())

def correct_and_replace_columns(user_query: str, actual_cols: list[str], threshold=80):
    """
    Fuzzy-matches potential column references in user query to actual column names,
    and returns a cleaned query + list of matched column names.
    """
    cleaned_actual = [clean_column_name(c) for c in actual_cols]
    matched = {}

    # Tokenize the user query into words
    words = re.findall(r'\w+', user_query)

    for word in words:
        cleaned = clean_column_name(word)
        match, score = process.extractOne(cleaned, cleaned_actual)
        if score >= threshold:
            matched[word] = actual_cols[cleaned_actual.index(match)]

    # Replace only whole words in query
    cleaned_query = user_query
    for original, corrected in matched.items():
        cleaned_query = re.sub(rf'\b{re.escape(original)}\b', corrected, cleaned_query)

    return cleaned_query, list(matched.values())
