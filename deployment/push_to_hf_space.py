"""
Hosting Script - Push deployment files to Hugging Face Space
Run this script to deploy the Streamlit app to Hugging Face Spaces
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file, login

# Configuration - UPDATE THESE VALUES
HF_TOKEN = os.environ.get("HF_TOKEN", "your_hf_token")
HF_USERNAME = "ansee178" 
SPACE_REPO = f"{HF_USERNAME}/wellness-tourism-app"

def main():
    # Login to Hugging Face
    login(token=HF_TOKEN)
    api = HfApi()
    
    # Create Space repository (Streamlit SDK)
    try:
        create_repo(
            repo_id=SPACE_REPO,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        print(f"✓ Space created: {SPACE_REPO}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Files to upload
    files = ["app.py", "requirements.txt", "Dockerfile"]
    
    for file_name in files:
        if os.path.exists(file_name):
            upload_file(
                path_or_fileobj=file_name,
                path_in_repo=file_name,
                repo_id=SPACE_REPO,
                repo_type="space",
                token=HF_TOKEN
            )
            print(f"✓ Uploaded: {file_name}")
    
    print(f"\n🚀 App deployed at: https://huggingface.co/spaces/{SPACE_REPO}")

if __name__ == "__main__":
    main()
