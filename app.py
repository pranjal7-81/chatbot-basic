import streamlit as st
import openai 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langchain Tracking

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with OpenAI"

# Prompt Template
prompt = ChatPromptTemplate([("system","You are a helpful assistant . Please respond to user queires"),("user","Question:{question}")])


def generate_response(question,api_key,llm,temprature,max_tokens):
    openai.api_key =  api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt|llm
    answer = chain.invoke({'question':question})
    return answer
    
    
