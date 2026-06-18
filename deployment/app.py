
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
    timeout=None,
    handle_parsing_errors = True)

api = HfApi(token=os.getenv("HF_TOKEN"))
DATABASE_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/customer_orders.db"

# Initializing SQL database object
db = SQLDatabase.from_uri("sqlite:///DATABASE_PATH")

# Defining a concise system message
system_message = """
Imagine you are an SQL assistant, expertized at performing sql queries.

Your role is to fetch relevant data from existing database 'db'.

Input is given as a query in simple text.

Output is the retrieved data from the database.
"""

# Initializing the SQL toolkit with customer database and pre-defined LLM
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

# Create the SQL agent with the system message
db_agent = create_sql_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = False,
#    handle_parsing_errors = True,
    system_message=SystemMessage(system_message)) 

# Defining a function to build a order query tool
def order_query(inputs) -> list:
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
    prompt = f"Generate one raw response related to: '{order_results}' considering the user query: '{user_query}'."
    response = llm.predict_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
    )
    raw_response = response.content.strip()
    return raw_response

order_query_tool = Tool(
    name = "OrderQueryTool",
    func = order_query,
    description = "Understands the context of an user query and match with order related information to generate a raw response.")

# Defining a function to refine raw responses into precise, polished answers to users
class Answering_Tool:
    def init(self, language_model = "en_core_web_sm"):
        """
        Initialize the Answering_Tool with a SpaCy language model.

        Args:
        - language_model(str): The SpaCy language model to use (default:"en_core_web_sm")
        """
        self.nlp=spacy.load(language_model)

    def polished_answer(self,raw_response):
        """
        Polish a raw response into a user-friendly answer.
        Args:
        - raw_response(str): The raw response to polish
        Returns: 
        - str: The polished answere
        """
        doc = self.nlp(response)
        polished_answer = raw_response + "."

        return polished_answer

answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.")

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
    # Defining a query response function to execute and run the built chat agent
    def query_response(self, order_id: str, user_query: str) -> str:
        # Fetch order information based on given order_id
#        order_details = db_agent.invoke(f"Fetch the order information related to Order ID '{order_id}' in a list")
        # Normalize to list
        if isinstance(order_details["output"], dict) and "items" in order_details["output"]:
            order_results = order_details["output"]["items"]
        elif isinstance(order_details["output"],str):
            order_results = [i.strip() for i in order_details["output"].split(",")]
        else:
            order_results = order_details["output"]
        # Agent Prompt
        agent_prompt = f"""
        The user querying for a particular order with Order ID, '{order_id}'.
        The user's query is '{user_query}'. 
        The details relevant to that particular order and user query are: '{order_results}'.
    
        Here is the process to follow:
        1. Confirm if the order information can match the expectation of an user query.
        2. If yes, generate a suitable response that address the query. If no, respond: "Sorry! Order not found."
        3. Pass the response generated in step: 2 to "Answering_Tool".
        4. Show the result got from the step: 3 as output.
        """
        response = chat_agent.run(agent_prompt)
        print(response)
        return response
    

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
#        response = self.query_response(
            order_id=self.order_id,
            user_query=user_query
        )

        return response
        
# Streamlit UI
st.title("🍔 FoodHub Delivery ChatBot")
st.write("Welcome to FoodHub Chat Support Assistant!")

# Session State Initialization
if "bot" not in st.session_state:
    st.session_state.bot = Chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Type your message here...")

# Chat Processing
if user_query:

    # Show User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Bot Response
#    response = st.session_state.bot.chat(user_query)

    # Store Assistant Response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response)
