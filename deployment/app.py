import json
import os 
import spacy
import sqlite3 
import getpass
import pandas as pd
import streamlit as st
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain import SQLDatabase, hub
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, AgentType, load_tools, Tool, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
import subprocess
import sys

#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 512,                                             # maximum number of tokens in the output
    max_retries=2,
    timeout=None)

def clone_repository(repo_url, local_path):
    try:
        subprocess.run(["git", "clone", repo_url, local_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
repo_url = "https://github.com/lokeshnathyadav-ctrl/capstone-project-26.git"
local_path = "./capstone-project-26"
if not os.path.exists(local_path):
    clone_repository(repo_url, local_path)
sys.path.insert(0, local_path)
#import my_functions
#st.write(my_functions.my_function())
from llm import llm
from db_tool import db_agent
from tools import order_query, order_query_tool, answer_query, answer_query_tool
from chat_agent import ChatBot, chat_agent
# Streamlit UI
st.title("🍔 FoodHub Delivery ChatBot")
st.write("Welcome to FoodHub Chat Support Assistant!")
if "bot" not in st.session_state:
    st.session_state.bot = Chatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:   # Display previous messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_query = st.chat_input("Type your message here...")    # User Input
if user_query:              # Chat Processing    
    st.session_state.messages.append(                                     # Chat Processing
        {
            "role": "user",
            "content": user_query
        }
    )
    with st.chat_message("user"):
        st.markdown(user_query)   
    response = st.session_state.bot.chat(user_query)    # Generate Bot Response
    # Store Assistant Response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )   
    with st.chat_message("assistant"):                      # Display Assistant Response
        st.markdown(response)
