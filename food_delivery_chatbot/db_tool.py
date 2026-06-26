import os
import json
import pandas as pd
import sqlite3
from langchain import hub
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict

#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 1024,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)
api = HfApi(token=os.getenv("HF_TOKEN"))
connection = sqlite3.connect("customer_orders.db")
DATASET_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/FoodHub_Go.csv"
df = pd.read_csv(DATASET_PATH)
df.to_sql('FoodHub_Go',connection,if_exists='append',index=False)
db = SQLDatabase.from_uri("sqlite:///customer_orders.db")
system_message = """
Below is an instruction that describes the task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Fetch and retrieve the order related details in the form of a JSON with every field in the fetched data beina a string.

###Input:
Order number/ID is given as input where the agent will fetch data related to Order ID from the provided database.

### Response:
"""
db_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type = "openai-tools",
    handle_parsing_errors = True,
    verbose = False,
    system_message= SystemMessage(system_message))
