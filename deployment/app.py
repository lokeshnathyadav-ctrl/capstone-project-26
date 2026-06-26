import subprocess
import sys
import os
import json
import pandas as pd
import sqlite3
import spacy
import sqlite3 
import getpass
import streamlit as st
from langchain import hub
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_agent, Tool, AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
#from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict

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
queryfunc_dir = os.environ['BUILD_DIR']
sys.path.insert(0,queryfunc_dir)
import queryfunc
from food_delivery_chatbot import chat_agent

# Streamlit UI
st.title("🍔 FoodHub Delivery ChatBot")
st.write("Welcome to FoodHub Chat Support Assistant!")
if "bot" not in st.session_state:
    st.session_state.bot = chat_agent.Chatbot()
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
