from langchain import SQLDatabase, hub
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, AgentType, load_tools, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
#from llm import llm
import os
import json
import pandas as pd
import sqlite3
#from db_tool import db_agent
#from tools import order_query, order_query_tool, answer_query, answer_query_tool

tools = [order_query_tool, answer_query_tool]

memory = ConversationBufferMemory(memory_key="chat_history")

chat_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True)

# Chatbot Class
class Chatbot:
    def __init__(self):
        self.chat_history = []
        self.order_id = None
        self.welcome_message = ("Hello! Welcome to Food Delivery Support 🍴")
        self.ask_order_message = ("Could you please share the Order ID you're searching for?")
    # Fetch Order Details
    def get_order_details(self,order_id):
        try:
            order_details = db_agent.invoke(f"Fetch the order information related to Order ID '{order_id}'")
            output = order_details.get("output",[])
            if isinstance(output,dict) and "items" in output:
                return output["items"]
            elif isinstance(output,str):
                return[i.strip() for i in output.split(",")]
            elif isinstance(output,list):
                return output
            else:
                return []
        except Exception as e:
            print(f"Database Error: {e}")
            return []
    # Defining a query response function to execute and run the built chat agent
    def query_response(self, order_id, user_query):
        order_results = self.get_order_details(order_id)
        if not order_results:
            return "Sorry! Order not found."     
        # Agent Prompt
        agent_prompt = f"""
        The user querying for a particular order with Order ID, '{order_id}'.

        The user's query is '{user_query}'. 
        The details relevant to that particular order and user query are: '{order_results}'.
    
        Here is the process to follow:
        1. Confirm if the order information can match the expectation of an user query. If true, go to step: 2 below:
        2. Generate a suitable raw response that appropriates the context present in the user query.
        3. Pass the response generated in step: 2 by passing it to 'PolishedResponses', where a concise and user-friendly response is generated.
        4. Reply the user with the generated response obtained in the step:3
        """
        try:
            response = chat_agent.run(agent_prompt)
            return response
        except Exception as e:
            
            return "Sorry! Something went wrong while processing your request."     
    # Main Chat Function
    def chat(self, user_query):
        self.chat_history.append(user_query)
        # First Interaction
        if len(self.chat_history) == 1:
            return (
                f"{self.welcome_message}\n\n"
                f"{self.ask_order_message}")
        # If order_id not captured yet
        if self.order_id is None:

            self.order_id = user_query.strip()
            return (
                f"Thanks! I see you shared your Order ID as "
                f"'{self.order_id}'.\n\n"
                f"Please tell me your concern with this order.")
        # Actual Query Processing 
        response = self.query_response(       
            order_id=self.order_id,
            user_query=user_query)
        return response   
