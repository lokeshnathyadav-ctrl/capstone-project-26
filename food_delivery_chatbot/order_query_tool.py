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

# Defining a function for order query tool
def order_query(inputs):
    """ 
    Takes the order context from the SQL agent and generate a raw response for the query
    Accepts a dict of order related information and the user's query.
    """
    # Handling a dict input
    if isinstance(inputs,dict):
        user_query = inputs.get("user_query", "")
        order_results = inputs.get("order_results", [])    
    else:
        # To Handle a string input fallback
        order_results = [i.strip() for i in str(inputs).split(",")]
        user_query = ""

    system_prompt = """
    You are an AI powered response generator. Your role is to generate a raw response for a given input.
    Input is a order context obtained from the SQL Agent.
    Understand the context of the provided input and frame a raw response as output.

    When crafting a raw response:
    1. Cross check if the order information is relevant to user query or not.
    2. Do not produce a raw response if the context provided to you doesn't match the user's query.
    3. In such case give output as "Invalid Request: Try again."

    """
    raw_responses = []
    for context in order_results:
        prompt = f"Generate one raw response related to: '{order_results}' considering the user query: '{user_query}'."
        response = llm.predict_messages(
        [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)])
        query = response.content.strip()
        if query:
            raw_responses.append(query)
        return raw_responses

        
# Defining a order query tool
order_query_tool = Tool(
    name = "OrderQueryTool",
    func = order_query,
    description = "Understands the context of an user query and match with order related information to generate a raw response.")
