import json          # to objectify Java script notion
import os            # to perform operating system related functions
import pandas as pd  # manipulating and working with data
import sqlite3       # Used to build SQL agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.agent_types import AgentType
from langchain import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain import hub
from langchain.agents import load_tools
from langchain.agents import Tool
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
# Supress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

groq_api_key = os.getenv('GROQ_API_KEY')

if groq_api_key:
    print("GROQ_API_KEY is set:", groq_api_key)
else:
    print("GROQ_API_KEY is not set in environment variables.")

llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 1024,                                              # maximum number of tokens in the output
    max_retries=2,
    api_key=groq_api_key,
    timeout=None)

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
