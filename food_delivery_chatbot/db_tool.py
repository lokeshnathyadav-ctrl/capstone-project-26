import os
import json
import pandas as pd
import sqlite3
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.utils.uuid import uuid7
#from langchain_openai import ChatOpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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
Give a response that appropriately completes the request.

### Instruction:
Fetch and retrieve the order related details.

###Input:
Order number or Order ID is given as input where the agent will fetch data related to this order from the database.

### Response: 
Retrieved order information in the format, specified in 'Ordercontext' class.
"""

@dataclass
class Ordercontext:
    order_id: str

@tool
def get_order_info(runtime: ToolRuntime[Ordercontext]) -> str:
    """Get the current user's order information."""
    order_id = runtime.context.order_id
    if order_id in db:
        order= db[order_id]
        return (
            f"Time of the Order: {order['order_time']}\n"
            f"Current status of the Order: {order['order_status']}\n"
            f"Payment Status: {order['payment_status']}\n"
            f"Food item ordered: {order['item_in_order']}\n"
            f"Time taken to prepare food: {order['preparing_eta']}\n"
            f"Time at which order prepared: {order['prepared_time']}\n"
            f"Estimated food delivery time: {order['delivery_eta']}\n"
            f"Time at which food is delivered: {order['delivery_time']}")
    return "Order not found"

db_agent = create_sql_agent(
    llm=llm,
    db=db,
    tools = [get_order_info],
    context_schema=Ordercontext,
#    agent_type = "openai-tools",
    handle_parsing_errors = True,
    verbose = False,
    system_message= SystemMessage(system_message))
