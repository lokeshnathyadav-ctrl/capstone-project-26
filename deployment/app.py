
# Importing required libraries to build a Chat UI
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# Function to get response from the chat agent
def get_reponse_from_chat_agent(query,chat_history=[]:
    response = chat_agent.run(agent_prompt)
    return response

# Initializing session state to keep track of chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]

st.title("FoodHub delivery ChatBot")

with st.form(key="message_form"):
    user_query=st.text_area("You:",height=100)
    submitted = st.form_submit_button(label='Send')
    
    if submitted:
        get_response_from_chat_agent(user_query)
