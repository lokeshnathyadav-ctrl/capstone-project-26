import json
import os 
import spacy
import sqlite3 
import getpass
import pandas as pd
import streamlit as st
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain import SQLDatabase, hub
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, AgentType, load_tools, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 512,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)

api = HfApi(token=os.getenv("HF_TOKEN"))
DATABASE_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/customer_orders.db"

# Initializing SQL database object
db = SQLDatabase.from_uri("sqlite:///DATABASE_PATH")

# Defining a concise system message
system_message = """Imagine you are an SQL assistant, expertized at performing sql queries.
Your role is to fetch data related to online food delivery from existing database.
Input is given as a query in simple text.
Output is the retrieved data from the database."""

# Initializing the SQL toolkit with customer database and pre-defined LLM
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

# Create the SQL agent with the system message
db_agent = create_sql_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = False,
    handle_parsing_errors = True,
    system_message=SystemMessage(system_message)) 
