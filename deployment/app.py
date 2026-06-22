import json
import os 
import spacy
import sqlite3 
import getpass
import requests
import pandas as pd
import streamlit as st
from huggingface_hub import login,HfApi
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
#from langchain.agents import create_agent
from langchain import SQLDatabase, hub
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, AgentType, load_tools, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
from huggingface_hub import login,HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 512,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)

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
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

db_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type = "openai-tools",
    handle_parsing_errors = True,
    verbose = False,
    system_message= SystemMessage(system_message))

# Defining a function to build a order query tool
def order_query(inputs):
    """ 
    Takes the order details as inputs and generates a raw response for the question put by the users.
    """
    if isinstance(inputs, dict):
        order_results = inputs.get("order_info", [])
        user_query = inputs.get("user_query","")
    else:
        order_results = [i.strip() for i in str(inputs).split(",")]
        user_query = ""

    system_prompt = """
    Below is an instruction that describes the task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Generate a raw response to the user, querying regard to his order related information. Understand the context present in the user query and generate a raw response by including the exact information he is querying for.  

    ### Input:
    Input is the JSON obtained from the output of 'db_agent'

    ### Response:
    A raw response which include the context of the user query paired with the necessary order information he's looking for.
    """
    
    prompt = """
    Generate one raw response related to the order '{order_id} and the user query '{user_query}'.

    While generating the raw reponse take help of these order related information to match the context present in user query with the appropriate order related information.
    Order Details:
    '{order_results}'
    """
    raw_response = llm.predict_messages([SystemMessage(content=system_prompt),HumanMessage(content=prompt)])
    return raw_response
       
order_query_tool = Tool(
    name = "OrderQueryTool",
    func = order_query,
    description = "Generates a raw response for the user query by including the appropriate retreived order information.")

# Defining a function to refine raw responses into precise, polished answers to users
nlp = spacy.load('en_core_web_sm')
def answer_query(raw_response):
    """
    Polishes a raw response by tokenizing, removing stop words and punctuation, 
    lemmatizing words, and reconstructing a simplified sentence.
    Args:
        raw_response (str): The input sentence obtained from 'OrderQueryTool' to be polished.
    Returns:
        str: The polished sentence meant to be replied to the user as response.
    """
    doc = nlp(raw_response)
    polished_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:                   # Check if the token is not a stop word and not punctuation
            polished_tokens.append(token.lemma_)                       # Lemmatize the token to its base form  
    polished_response = ' '.join(polished_tokens)                     # Reconstruct the polished sentence
    return polished_response

answer_query_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Polishes the raw response which are obtained from calling the 'OrderQueryTool' into precise, clear and user-friendly responses.")

# Initialize Tools & Agent
tools = [order_query_tool, answer_query_tool]

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
        self.welcome_message = ("Hello! Welcome to Food Delivery Support 🍴")
        self.ask_order_message = ("Could you please share the Order ID you're searching for?")
    # Fetch Order Details
    def get_order_details(self,order_id):
        try:
            order_details = dg_agent.invoke(f"Fetch the order information related to Order ID '{self.order_id}'")
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
        The user querying for a particular order with Order ID, '{self.order_id}'.
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
            return "Sorry! Something went wron while processing your request."
    
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
                f"Please tell me your concern regarding the order.")
        # Actual Query Processing 
        response = self.query_response(       
            order_id=self.order_id,
            user_query=user_query)
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
    response = st.session_state.bot.chat(user_query)   
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
