"""
Data Preparation Script
Cleans data, splits into train/test, and uploads to Hugging Face
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, upload_file
from datasets import load_dataset
import joblib

def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")
    DATASET_REPO = f"{HF_USERNAME}/wellness-tourism-dataset"
    
    login(token=HF_TOKEN)
    
    # Load data from Hugging Face
    dataset = load_dataset(DATASET_REPO, data_files="tourism.csv", split="train")
    df = dataset.to_pandas()
    
    # Clean data
    # Remove unnecessary columns
    cols_to_drop = ['CustomerID']
    if 'Unnamed: 0' in df.columns:
        cols_to_drop.append('Unnamed: 0')
    first_col = df.columns[0]
    if first_col not in ['ProdTaken', 'Age'] and df[first_col].dtype in ['int64', 'float64']:
        cols_to_drop.append(first_col)
    
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Handle missing values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    df = df.drop_duplicates()
    
    # Encode categorical variables
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Split data
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save and upload
    os.makedirs("tourism_project/data", exist_ok=True)
    os.makedirs("tourism_project/model_building", exist_ok=True)
    
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)
    joblib.dump(encoders, "tourism_project/model_building/label_encoders.joblib")
    
    # Upload to HF
    for file_name in ["train.csv", "test.csv"]:
        upload_file(
            path_or_fileobj=f"tourism_project/data/{file_name}",
            path_in_repo=file_name,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
    
    print("✓ Data preparation completed!")

if __name__ == "__main__":
    main()
