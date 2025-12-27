"""
Deployment Script
Pushes app files to Hugging Face Space
"""

import os
from huggingface_hub import login, create_repo, upload_file

def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")
    SPACE_REPO = f"{HF_USERNAME}/wellness-tourism-app"
    
    login(token=HF_TOKEN)
    
    # Create Space
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="streamlit", exist_ok=True)
    
    # Upload files
    files = [
        ("tourism_project/deployment/app.py", "app.py"),
        ("tourism_project/deployment/requirements.txt", "requirements.txt"),
        ("tourism_project/deployment/Dockerfile", "Dockerfile")
    ]
    
    for local_path, repo_path in files:
        if os.path.exists(local_path):
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=SPACE_REPO,
                repo_type="space",
                token=HF_TOKEN
            )
            print(f"✓ Uploaded: {repo_path}")
    
    print(f"🚀 Deployed at: https://huggingface.co/spaces/{SPACE_REPO}")

if __name__ == "__main__":
    main()
