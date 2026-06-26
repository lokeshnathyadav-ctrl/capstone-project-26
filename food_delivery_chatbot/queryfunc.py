import os
import json
import pandas as pd
#import sqlite3
import spacy
#from langchain_community.utilities import SQLDatabase
#from langchain_groq import ChatGroq
#from langchain_community.agent_toolkits.sql.base import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
#from langchain.agents.agent_types import AgentType
from langchain.tools import tool
#from langchain_community.agent_toolkits.load_tools import load_tools
#from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
@tool("response_generator", description="Generates a raw response to the user query by identifying the results of an order.")
def order_query(inputs):
    """ 
    Takes the order details as inputs and generates a raw response for the question put by the users.
    """
    if isinstance(inputs, dict):        
        order_results = inputs.get("order_details", [])
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
    
    prompt = f"""
    Generate one raw response related to the order '{order_id}' and the user query '{user_query}'.

    While generating the raw reponse take help of these order related information to match the context present in user query with the appropriate order related information.
    Order Details: '{order_results}'
    """
    raw_response = llm.predict_messages([SystemMessage(content=system_prompt),HumanMessage(content=prompt)])
    return raw_response

@tool("response_formalizer", description = "Polishes the raw response obtained in 'order_query' tool into precise, clear and user-friendly reply message.")
def answer_query(raw_response):
    """
    Polishes a raw response by tokenizing, removing stop words and punctuation, 
    lemmatizing words, and reconstructing a simplified sentence.
    Args:
        raw_response (str): The input sentence obtained from 'OrderQueryTool' to be polished.
    Returns:
        str: The polished sentence meant to be replied to the user as response.
    """
    nlp = spacy.load('en_core_web_sm')
#    nlp = en_core_web_sm.load()
    doc = nlp(raw_response)
    polished_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:                   # Check if the token is not a stop word and not punctuation
            polished_tokens.append(token.lemma_)                       # Lemmatize the token to its base form  
    polished_response = ' '.join(polished_tokens)                     # Reconstruct the polished sentence
    return polished_response
