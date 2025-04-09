# Chatbot Demo Examples

This directory contains test cases demonstrating the chatbot's capabilities.

## File Structure
- `chatbot_demo.json`: Test cases with prompts and expected responses
- `sample_sales.csv`: Sample data file for analysis tests

## How to Test

1. For name handling:
```bash
curl -X POST http://localhost:8000/api/chat \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{"prompt":"My name is Alex"}'
```

2. For data analysis:
```bash
curl -X POST http://localhost:8000/api/chat \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{"prompt":"Analyze this sales data"}' \
-F "file=@sample_sales.csv"
```

3. For general knowledge:
```bash
curl -X POST http://localhost:8000/api/chat \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{"prompt":"Explain quantum computing"}'
```

## Expected Behavior
- Name handling: Stores and recalls names
- Data analysis: Provides statistical insights
- General knowledge: Falls back to LLM when no tools apply

## Sample Data
The included `sample_sales.csv` contains:
- Date, Product, Price, Units Sold, Region
- 100 sample transactions across different regions

## Automated Testing
To run automated tests:
```bash
python -m pytest tests/ -v
