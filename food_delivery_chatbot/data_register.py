# Installing required libraries
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os
import requests

# User credentials to access HF Hub
repo_id = "Lokeshnathy/foodhub-orders-data"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Checking if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
    print(f"Space '{repo_id}' created.")

# Uploads the data to Hugging Face Hub 
api.upload_folder(
    folder_path="D:/Capstone_Project_26/data",
    repo_id=repo_id,
    repo_type=repo_type)    
