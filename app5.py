import streamlit as st

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from langfuse.langchain import CallbackHandler

# =====================================
# Load Environment Variables
# =====================================

import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# Langfuse Keys
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

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
    "Enter an expense description and get category, type, and confidence note."
)

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

class ExpenseCategoryOutput(BaseModel):
    expense_category: str = Field(
        description="Expense category"
    )


class ExpenseTypeOutput(BaseModel):
    expense_type: str = Field(
        description="Essential, Non-Essential, Uncertain"
    )


class ConfidenceOutput(BaseModel):
    confidence_note: str = Field(
        description="Reasoning note"
    )

# =====================================
# Parsers
# =====================================

category_parser = PydanticOutputParser(
    pydantic_object=ExpenseCategoryOutput
)

type_parser = PydanticOutputParser(
    pydantic_object=ExpenseTypeOutput
)

confidence_parser = PydanticOutputParser(
    pydantic_object=ConfidenceOutput
)

# =====================================
# Category Prompt
# =====================================

category_prompt = ChatPromptTemplate.from_template(
"""
Classify the expense into ONE category.

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
    "format_instructions":
    category_parser.get_format_instructions()
}
)

category_chain = (
    category_prompt
    | llm
    | category_parser
    | RunnableLambda(
        lambda x: x.expense_category
    )
)

# =====================================
# Expense Type Prompt
# =====================================

type_prompt = ChatPromptTemplate.from_template(
"""
Determine spending type.

Labels:
- Essential
- Non-Essential
- Uncertain

Rules:

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

            st.success("Expense Categorized Successfully!")

            st.subheader("JSON Output")

            st.json(result)

            st.subheader("Results")

            st.write(
                f"**Expense Category:** {result['expense_category']}"
            )

            st.write(
                f"**Expense Type:** {result['expense_type']}"
            )

            st.write(
                f"**Confidence Note:** {result['confidence_note']}"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")