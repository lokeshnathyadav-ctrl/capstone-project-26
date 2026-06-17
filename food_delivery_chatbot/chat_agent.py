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

# Initialize Tools & Agent
tools = [order_query_tool, answer_tool]

# Defining the memory for 'conversational_react_description" agent type
memory = ConversationBufferMemory(memory_key="chat_history")

chat_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True)

# Chatbot Class
class Chatbot:

    def __init__(self):

        self.chat_history = []
        self.order_id = None

        self.welcome_message = (
            "Hello! Welcome to Food Delivery Support 🍴"
        )

        self.ask_order_message = (
            "Could you please share the Order ID you're searching for?"
        )

    # Fetch Order Details
   
    def get_order_details(self, order_id):

        try:

            order_details = db_agent.invoke(
                f"Fetch the order information related to "
                f"Order ID '{order_id}' in a list"
            )

            output = order_details.get("output", [])

            # Normalize outputs
            if isinstance(output, dict) and "items" in output:
                return output["items"]

            elif isinstance(output, str):
                return [i.strip() for i in output.split(",")]

            elif isinstance(output, list):
                return output

            else:
                return []

        except Exception as e:
            print(f"Database Error: {e}")
            return []

    # Generate Agent Response
    def query_response(self, order_id:str, user_query:str) -> str:

        order_results = self.get_order_details(order_id)

        agent_prompt = f"""
        The user is querying for an order with Order ID '{order_id}'.
        The 'db_agent' successfully retrieved the Order Details from the database.

        User Query:
        '{user_query}'

        Order Details:
        '{order_results}'

        Instructions:
        1. Understand the user's concern related to the order.
        2. Use the order details to generate a raw response to a user query using the 'OrderQueryTool'.
        3. Later polish the gererated raw response using 'PolishedResponses' tool and pass the final response as reply to the user query. 
        4. If the order is unavailable, respond politely using the 'PolishedResponses' tool and pass the response as reply to the user query.
        """

        try:
            response = chat_agent.run(agent_prompt)
            return response

        except Exception as e:
            print(f"Agent Error: {e}")
            return "Sorry! Something went wrong while processing your request."

   # Main Chat Function
    def chat(self, user_query):

        self.chat_history.append(user_query)

        # First Interaction
        if len(self.chat_history) == 1:

            return (
                f"{self.welcome_message}\n\n"
                f"{self.ask_order_message}"
            )

        # If order_id not captured yet
        if self.order_id is None:

            self.order_id = user_query.strip()

            return (
                f"Thanks! I found your Order ID as "
                f"'{self.order_id}'.\n\n"
                f"Please tell me your concern regarding the order."
            )

        # Actual Query Processing
        response = self.query_response(
            order_id=self.order_id,
            user_query=user_query
        )

        return response
