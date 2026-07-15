# ================================
# Expense Categorizer App
# ================================

import streamlit as st

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from langfuse.langchain import CallbackHandler

import os
from dotenv import load_dotenv

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

if not langfuse_public_key or not langfuse_secret_key:
    st.error("Langfuse keys not found in .env file")
    st.stop()

# =====================================
# Langfuse
# =====================================

langfuse_handler = CallbackHandler()

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
page_title="Expense Categorizer",
page_icon="💰",
layout="centered"
)

st.title("💰 Expense Categorizer")

st.write(
"Enter an expense description and get amount, category, type, and analytics."
)

# =====================================
# Session State
# =====================================

if "expenses" not in st.session_state:
    st.session_state.expenses = []

# =====================================
# LLM
# =====================================

llm = ChatGroq(
model="qwen/qwen3-32b",
groq_api_key=groq_api_key,
reasoning_format="hidden"
)

# =====================================
# Pydantic Models
# =====================================

class AmountOutput(BaseModel):
    amount: float = Field(
        description="Expense amount. Return 0 if not found.")

from typing import List

class ExpenseItem(BaseModel):
    description: str = Field(description="Expense description")
    amount: float = Field(description="Expense amount")
    expense_category: str = Field(description="Expense category")

class ExpenseCategoryOutput(BaseModel):
    expenses: List[ExpenseItem]

class ExpenseTypeOutput(BaseModel):expense_type: str = Field(description="Essential, Non-Essential, Uncertain")

class ConfidenceOutput(BaseModel):confidence_note: str = Field(description="Reasoning note")

# =====================================
# Parsers
# =====================================

amount_parser = PydanticOutputParser(pydantic_object=AmountOutput)

category_parser = PydanticOutputParser(pydantic_object=ExpenseCategoryOutput)

type_parser = PydanticOutputParser(pydantic_object=ExpenseTypeOutput)

confidence_parser = PydanticOutputParser(pydantic_object=ConfidenceOutput)

# =====================================
# Amount Prompt
# =====================================

amount_prompt = ChatPromptTemplate.from_template(
"""
Extract the expense amount.

Rules:

* Return only numeric amount.
* Ignore currency symbols such as ₹, $, etc.
* Return 0 if amount is not present.

Expense:
{expense_text}

{format_instructions}
""",
partial_variables={"format_instructions":amount_parser.get_format_instructions()})

amount_chain = (amount_prompt| llm| amount_parser| RunnableLambda(lambda x: x.amount))

# =====================================
# Category Prompt
# =====================================

category_prompt = ChatPromptTemplate.from_template(
"""
Extract EVERY expense separately.

For each expense return:

- description
- amount
- expense_category

Categories:

- Food & Groceries
- Transportation
- Utilities
- Housing
- Healthcare
- Entertainment
- Shopping
- Education
- Travel
- Miscellaneous

Expense:

{expense_text}

{format_instructions}
""",
partial_variables={
"format_instructions":category_parser.get_format_instructions()
})

category_chain = (category_prompt| llm| category_parser| RunnableLambda(lambda x: x.expenses))
# =====================================
# Type Prompt
# =====================================

type_prompt = ChatPromptTemplate.from_template(
"""
Determine spending type.

Labels:

* Essential
* Non-Essential
* Uncertain

Essential:
Food, groceries, rent,
electricity bill, fuel,
medical expenses,
education fees.

Non-Essential:
Luxury items,
gadgets,
entertainment,
premium subscriptions.

Uncertain:
Insufficient information.

Expense:
{expense_text}

{format_instructions}
""",
partial_variables={
"format_instructions":
type_parser.get_format_instructions()
}
)

type_chain = (
type_prompt
| llm
| type_parser
| RunnableLambda(
lambda x: x.expense_type
)
)

# =====================================

# Confidence Prompt

# =====================================

confidence_prompt = ChatPromptTemplate.from_template(
"""
Generate a confidence note.

Rules:

1. Explain classification briefly.
2. Maximum one sentence.
3. Mention uncertainty if needed.

Expense:
{expense_text}

{format_instructions}
""",
partial_variables={
"format_instructions":
confidence_parser.get_format_instructions()
}
)

confidence_chain = (
confidence_prompt
| llm
| confidence_parser
| RunnableLambda(
lambda x: x.confidence_note
)
)

# =====================================

# Parallel Chain

# =====================================

expense_analysis = RunnableParallel(
{
"amount": amount_chain,
"expense_category": category_chain,
"expense_type": type_chain,
"confidence_note": confidence_chain
}
)

# =====================================

# User Input

# =====================================

expense_text = st.text_area(
"Enter Expense Description"
)

# =====================================

# Button

# =====================================

if st.button("Categorize Expense"):

    if not expense_text.strip():
        st.warning("Please enter an expense description.")

    else:
        try:
            with st.spinner("Analyzing expense..."):

                result = expense_analysis.invoke(
                    {
                        "expense_text": expense_text
                    },
                    config={
                        "callbacks": [langfuse_handler]
                    }
                )

            st.session_state.expenses.append(result)

            amount = result["amount"]

            if amount < 500:
                spending_level = "Low"
            elif amount < 2000:
                spending_level = "Medium"
            else:
                spending_level = "High"

            total_expenses = sum(
                item["amount"]
                for item in st.session_state.expenses
            )

            category_totals = {}

            for expense in result["expense_category"]:

                category = expense.expense_category
                amount = expense.amount

                if category not in category_totals:
                    category_totals[category] = 0

                category_totals[category] += amount


            st.success("Expense Categorized Successfully!")

            st.subheader("JSON Output")
            st.json(result)

            st.subheader("Results")

            st.write(f"**Total Amount:** ₹{total_expenses}")

            st.subheader("Detected Expenses")

            for expense in result["expense_category"]:
                st.write(
                    f"• {expense.description} | ₹{expense.amount} | {expense.expense_category}"
                )
            st.write(
                f"**Expense Type:** {result['expense_type']}"
            )

            st.write(
                f"**Confidence Note:** {result['confidence_note']}"
            )

            st.write(
                f"**Spending Level:** {spending_level}"
            )

            st.metric(
                "Expenses cast",
                f"₹{total_expenses:,.2f}"
            )

            st.subheader("Category Summary")

            for category, total in category_totals.items():

                st.write(
                    f"**{category}:** ₹{total:,.2f}" )
            total_expenses = sum(
                    expense.amount
                    for expense in result["expense_category"]
                )
            st.metric(
                "Total Expenses",f"₹{total_expenses:,.2f}")

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )