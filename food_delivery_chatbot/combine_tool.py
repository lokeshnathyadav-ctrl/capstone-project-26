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
import getpass
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

order_query_tool = Tool(
    name = "OrderQueryTool",
    func = order_query,
    description = "Understands the context of an user query and match with order related information to generate a raw response.")
answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.")

# Initialize Tools & Agent
tools = [order_query_tool, answer_tool]

chat_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True
)
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
  
    def query_response(self, order_id, user_query):

        order_results = self.get_order_details(order_id)

        if not order_results:
            return "Sorry! Order not found."

        agent_prompt = f"""
        The user is querying for an order with Order ID '{order_id}'.

        User Query:
        '{user_query}'

        Order Details:
        '{order_results}'

        Instructions:
        1. Understand the user's concern.
        2. Use the order details to answer accurately.
        3. If the order is unavailable, respond politely.
        4. Pass the final response through 'Answering_Tool'.
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
