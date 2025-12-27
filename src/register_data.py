"""
Data Registration Script
Uploads the tourism dataset to Hugging Face Dataset Space
"""

import os
from huggingface_hub import login, create_repo, upload_file

def main():
    # Get credentials from environment
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")
    DATASET_REPO = f"{HF_USERNAME}/wellness-tourism-dataset"
    
    # Login
    login(token=HF_TOKEN)
    
    # Create repository
    try:
        create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
        print(f"✓ Dataset repo created: {DATASET_REPO}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload data
    upload_file(
        path_or_fileobj="tourism_project/data/tourism.csv",
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("✓ Data uploaded successfully!")

if __name__ == "__main__":
    main()
