
# Importing necessary libraries
from huggingface_hub import HfApi
import os

# Define user credentials for HF Hub
api=HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Lokeshnathy/foodhub-delivery-chatbot"
repo_type="space"

# Uploads the required deployment files to HF Hub
api.upload_folder(
    folder_path = "D:/Capstone_Project_26/deployment", 
    repo_id = repo_id,
    repo_type = repo_type)
