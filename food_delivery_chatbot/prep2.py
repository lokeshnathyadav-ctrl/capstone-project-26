import os
import sqlite3
import pandas as pd
import requests
from huggingface_hub import login,HfApi
api = HfApi(token=os.getenv("HF_TOKEN"))
connection = sqlite3.connect("customer_orders.db")
DATASET_PATH = "hf://datasets/Lokeshnathy/foodhub-orders-data/FoodHub_Go.csv"
df = pd.read_csv(DATASET_PATH)
df.to_sql('FoodHub_Go',connection,if_exists='append',index=False)
