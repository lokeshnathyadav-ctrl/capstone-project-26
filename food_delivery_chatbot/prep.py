import pandas as pd
import os
import sqlite3
import requests
from huggingface_hub import login,HfApi
api = HfApi(token=os.getenv("HF_TOKEN"))
# Defining the path of the uploaded data in Hugging Face Hub
DATABASE_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/customer_orders.db"
print("Connected to database")
ORDERS_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/orders_table.csv"
CONSUMERS_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/consumers_table.csv"
DELIVERY_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/delivery_table.csv"
print("Dataset loaded successfully")

# Appending the data with the provided 'customer_orders' database
orders_df = pd.read_csv(ORDERS_PATH)
consumers_df = pd.read_csv(CONSUMERS_PATH)
delivery_df = pd.read_csv(DELIVERY_PATH)
orders_df.to_sql('orders_table',connection,if_exists='append',index=False)
consumers_df.to_sql('consumers_table',connection,if_exists='append',index=False)
delivery_df.to_sql('delivery_table',connection,if_exists='append',index=False)
# Establishing a connection with the provided database.
connection = sqlite3.connect("DATABASE_PATH")
