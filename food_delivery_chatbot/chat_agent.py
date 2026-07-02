import os
import json
import pandas as pd
import sqlite3
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
from queryfunc import order_query, answer_query
from langchain.agents import create_agent
import re
from dataclasses import dataclass
from langchain.messages import ToolMessage


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
chatagent = create_agent(
    tools=[order_query, answer_query],
    model=llm,
    checkpointer=InMemorySaver(),
    context_schema=Ordercontext,
    system_prompt="You are an online food delivery application's support assistant." 
)
class Chatbot:                                                      # Chatbot Class
    def __init__(self):
        self.config = []
        self.order_id = None
        self.welcome_message = ("Hello! Welcome to Food Delivery Support 🍴")
        self.ask_order_message = ("Could you please share the Order ID you're searching for?")
    def get_order_results(self, order_id):
        if order_id.isdigit() and len(order_id) >= 5:
            return order_id
        if re.match(r'^O\d+$', order_id):
            return order_id
        return f"'{order_id}' is not a valid Order ID."
        try:
            order_results = db_agent.invoke({"messages": [{"role": "user", "content": "Fetch the data present in all the columns"}]}, 
                                            config = {"configurable": {"thread_id": str(uuid7())}}
                                            ,context=Ordercontext(order_id))
            return response
        except Exception as e:
            print(f"Error getting order results: {str(e)}")
            return None         
    def query_response(self, order_id, user_query):
        order_results = self.get_order_results(order_id) # 'Chatbot' object has no attribute 'get_order_details'
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
            response = chatagent.invoke(agent_prompt)["messages"][-1].content       
            return response        
        except Exception as e:          
            return "Sorry! Something went wrong while processing your request."     
    def chat(self, user_query):
        self.config.append(user_query)
        if len(self.config) == 1:                                 # First Interaction
            return (
                f"{self.welcome_message}\n\n"
                f"{self.ask_order_message}")
        if self.order_id is None:                                       # If order_id not captured yet
            self.order_id = user_query.strip()            
            return (               
                f"Thanks! I see you shared your Order ID as "                
                f"'{self.order_id}'.\n\n"
                f"Please tell me your concern with this order.")                
        response = self.query_response(                                                   # Actual Query Processing 
            order_id=self.order_id,
            user_query=user_query)
        return response 
