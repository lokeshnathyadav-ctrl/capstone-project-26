import os
import json
import pandas as pd
import sqlite3
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
from queryfunc import order_query, answer_query

llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 1024,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)
config = {"configurable": {"thread_id": str(uuid7())}}
chat_agent = create_agent(
    tools=[order_query, answer_query],
    llm=llm,
    checkpointer=InMemorySaver())
# Chatbot Class
class Chatbot:
    def __init__(self):
        self.config = []
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
            print(f"Database Error: {e}")                 # Assuming code passed here!
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
            response = chat_agent.invoke(agent_prompt,config=config)
            return response
        except Exception as e:          
            return "Sorry! Something went wrong while processing your request."     

    # Main Chat Function
    def chat(self, user_query):
        self.config.append(user_query)
        # First Interaction
        if len(self.config) == 1:
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
