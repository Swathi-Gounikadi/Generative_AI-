# Expense Classification and Spending Analysis using LangChain, Groq, and Langfuse

## Overview

This project demonstrates how to build a structured expense analysis system using Large Language Models (LLMs), LangChain, Groq, and Langfuse.

The system automatically analyzes an expense description and generates:

1. Expense Category
2. Spending Type
3. Confidence Note

The application uses multiple parallel LLM chains and returns structured outputs using Pydantic models.

---

## Features

### Expense Category Classification

Classifies expenses into one of the following categories:

* Food & Groceries
* Transportation
* Utilities
* Housing
* Healthcare
* Entertainment
* Shopping
* Education
* Travel
* Miscellaneous

### Spending Type Detection

Determines whether an expense is:

* Essential
* Non-Essential
* Uncertain

### Confidence Explanation

Generates a short explanation describing why the classification was assigned.

### Parallel Execution

Uses LangChain's RunnableParallel to execute multiple LLM chains simultaneously, improving efficiency and reducing latency.

### Observability with Langfuse

Tracks:

* Prompts
* Model Calls
* Responses
* Latency
* Execution Traces

for monitoring and debugging LLM workflows.

---

## Architecture

Input Expense Description
↓
RunnableParallel
├── Category Chain
├── Spending Type Chain
└── Confidence Note Chain
↓
Structured JSON Output

---

## Technology Stack

* Python
* LangChain
* Groq API
* Langfuse
* Pydantic
* Google Colab

---

## Model Used

Model Provider: Groq

Model Name:

qwen/qwen3-32b

Reasoning Format:

hidden

---

## Project Workflow

### Step 1: User Inputs Expense

Example:

Paid monthly electricity bill

### Step 2: Category Classification Chain

Output:

Utilities

### Step 3: Spending Type Classification Chain

Output:

Essential

### Step 4: Confidence Note Chain

Output:

Electricity bills are necessary household utility expenses.

### Step 5: Parallel Aggregation

Final Result:

```python
{
    "expense_category": "Utilities",
    "expense_type": "Essential",
    "confidence_note": "Electricity bills are necessary household utility expenses."
}
```

---

## Sample Results

### Example 1

Input:

```text
Paid monthly electricity bill
```

Output:

```python
{
    "expense_category": "Utilities",
    "expense_type": "Essential",
    "confidence_note": "Electricity bills are necessary household utility expenses."
}
```

---

### Example 2

Input:

```text
Bought groceries from supermarket
```

Output:

```python
{
    "expense_category": "Food & Groceries",
    "expense_type": "Essential",
    "confidence_note": "Groceries are basic household necessities."
}
```

---

### Example 3

Input:

```text
Bought premium smartwatch
```

Output:

```python
{
    "expense_category": "Shopping",
    "expense_type": "Non-Essential",
    "confidence_note": "A premium smartwatch is generally considered discretionary spending."
}
```

---

## LangChain Components Used

### ChatPromptTemplate

Used to create structured prompts for each task.

### PydanticOutputParser

Ensures outputs follow a predefined schema.

### RunnableLambda

Extracts desired fields from parsed outputs.

### RunnableParallel

Runs multiple chains concurrently.

### ChatGroq

Provides access to Groq-hosted LLMs.

---

## Langfuse Monitoring

Langfuse is integrated using:

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()
```

Benefits:

* Prompt tracking
* Trace visualization
* Performance monitoring
* Error debugging
* Cost analysis

---

## Project Structure

```text
expense-analysis/
│
├── category_chain.py
├── type_chain.py
├── confidence_chain.py
├── models.py
├── prompts.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Multi-label expense classification
* Budget tracking
* Expense anomaly detection
* Monthly spending summaries
* Personal finance dashboard
* Vector database integration
* Agent-based financial advisor

---

## Learning Outcomes

By completing this project, you will understand:

* Structured LLM Outputs
* Pydantic Output Parsing
* LangChain Runnables
* Parallel Chain Execution
* Prompt Engineering
* LLM Observability with Langfuse
* Production-ready AI Workflow Design

---

## Conclusion

This project showcases a practical real-world application of Large Language Models for personal finance analysis. By combining LangChain, Groq, Pydantic, and Langfuse, the system delivers structured, reliable, and observable expense classifications suitable for intelligent finance applications.
