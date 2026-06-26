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

#app_dir = os.path.dirname(os.path.abspath('app.py'))

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
#import db_tool
#st.write(db_tool.db_agent
#from food_delivery_chatbot import db_tool, queryfunc, chat_agent 
#st.write(tools.order_query())
#st.write(tools.answer_query())
#import chat_agent
#st.write(chat_agent.Chatbot())
#from llm import llm
#from db_tool import db_agent, llm
#from tools import order_query, order_query_tool, answer_query, answer_query_tool
#from chat_agent import ChatBot, chat_agent
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
