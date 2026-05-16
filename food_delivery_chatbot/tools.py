
import json          # to objectify Java script notion
import os            # to perform operating system related functions
import pandas as pd  # manipulating and working with data
import sqlite3       # Used to build SQL agent

# Langchain framework to create custom agents for open-source LLMs
from langchain.agents import create_sql_agent, initialize_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import load_tools
from langchain.agents import Tool
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict

# Supress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# To import the LLM
from langchain_groq import ChatGroq
import getpass
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
#from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"

llm = ChatGroq(
    model = "meta-llama/llama-4-scout-17b-16e-instruct",           # Name of the chat model
    temperature = 0,                                               # Temperature setting to '0', for consistent and deterministic responses
    max_tokens = 1024,                                              # maximum number of tokens in the output
    max_retries=2,
    timeout=None)

DATABASE_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/customer_orders.db"

# Initializing SQL database object
db = SQLDatabase.from_uri("sqlite:///DATABASE_PATH")

# Defining a concise system message
system_message = """Imagine you are an SQL assistant, expertized at performing sql queries.
Your role is to fetch data related to online food delivery from existing database.
Input is given as a query in simple text.
Output is the retrieved data from the database."""

# Initializing the SQL toolkit with customer database and pre-defined LLM
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

# Create the SQL agent with the system message
db_agent = create_sql_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = False,
    handle_parsing_errors = True,
    system_message=SystemMessage(system_message))

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

# Defining a function to refine raw responses into precise, polished answers to users
class Answering_Tool:
    def init(self, language_model = "en_core_web_sm"):
        """
        Initialize the Answering_Tool with a SpaCy language model.

        Args:
        - language_model(str): The SpaCy language model to use (default:"en_core_web_sm")
        """
        self.nlp=spacy.load(language_model)

    def polish_answers(self,raw_response):
        """
        Polish a raw response into a user-friendly answer.

        Args:
        - raw_response(str): The raw response to polish

        Returns:
        - str: The polished answer
        """
        # Process the raw response with SpaCy
        doc = self.nlp(raw_response)

        # Perform polishing tasks, such as:
        # - Removing stop words
        # - Lemmatizing words
        # - Shortening sentences

        # For demonstration purposes, simply return the raw response with proper punctuation
        polished_answer = raw_response + "."

        return polished_answer

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

        # For demonstration purposes, simply return the polished raw response
        polished_answer = self.polish_answer(raw_response)

        return polished_answer

answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.") 

# Defining the memory for 'conversational_react_description" agent type
memory = ConversationBufferMemory(memory_key="chat_history")

# Integrating order query tool & answer tool
tools = [order_query_tool,
         answer_tool]

# Defining the food delivery chat agent
chat_agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose = False,
    memory=memory,
    handle_parsing_errors = True)

# To implement the interactive chat loop
# Takes order input
# Fetch order info
# Clubs the user query along with order info to generate raw response
# Polishes the raw response into polished user-friendly replies

def query_response(order_id: str, user_query: str) -> str:
    # Fetch order information based on given order_id
    order_details = db_agent.invoke(f"Fetch the order information related to Order ID '{order_id}' in a list")

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
