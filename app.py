from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()


# =========================
# LangSmith Tracking
# =========================

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OLLAMA"


# =========================
# Prompt Template
# =========================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Please respond to user queries."
    ),
    (
        "user",
        "Question: {question}"
    )
])


# =========================
# Generate Response
# =========================

def generate_response(question, model, temperature, max_tokens):

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    answer = chain.invoke({
        "question": question
    })

    return answer


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Simple Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Simple Q&A Chatbot With Ollama")

st.write(
    "Ask anything and get a response from your local Ollama model."
)


# =========================
# Sidebar
# =========================

st.sidebar.header("Settings")

model = st.sidebar.selectbox(
    "Select Ollama Model",
    ["gemma:latest"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)


# =========================
# User Input
# =========================

question = st.text_input(
    "Ask your question:",
    placeholder="What is artificial intelligence?"
)


# =========================
# Generate Response
# =========================

if st.button("Generate Response"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                response = generate_response(
                    question,
                    model,
                    temperature,
                    max_tokens
                )

                st.subheader("Response")

                st.write(response)

            except Exception as e:

                st.error(f"Error: {e}")