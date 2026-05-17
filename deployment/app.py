import streamlit as st
from langchain.agents import initialize_agent, AgentType
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

import getpass
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 1024,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)

# Defining a function to build a order query tool
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

    def generate_answer(self,question,raw_response):
        """
        Generate a user-friendly answer based on a question and a raw response.

        Args:
        - question (str): The question being asked
        - raw_response (str): The raw response to polish

        Returns
        - str: The polished answer
        """
        # Process the question and raw response with Spacy
        question_doc = self.nlp(question)
        raw_response_doc = self.nlp(raw_response)

        # Perform answer generation tasks, such as:
        # - Identifying relevant entities
        # - Determining the answer's sentiment
        polished_answer = self.polish_answer(raw_response)

        return polished_answer

answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.")


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
    handle_parsing_errors=True
)
# Streamlit UI
st.set_page_config(
    page_title="FoodHub Delivery ChatBot",
    page_icon="🍔"
)

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
