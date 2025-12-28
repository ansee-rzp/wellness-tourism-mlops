"""
Model Training Script
Trains models with MLflow tracking and uploads best model to Hugging Face
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import mlflow
import joblib
from huggingface_hub import login, create_repo, upload_file
from datasets import load_dataset

def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")
    DATASET_REPO = f"{HF_USERNAME}/wellness-tourism-dataset"
    MODEL_REPO = f"{HF_USERNAME}/wellness-tourism-model"
    
    login(token=HF_TOKEN)
    
    # Load data
    train_data = load_dataset(DATASET_REPO, data_files="train.csv", split="train").to_pandas()
    test_data = load_dataset(DATASET_REPO, data_files="test.csv", split="train").to_pandas()
    
    X_train = train_data.drop('ProdTaken', axis=1)
    y_train = train_data['ProdTaken']
    X_test = test_data.drop('ProdTaken', axis=1)
    y_test = test_data['ProdTaken']
    
    # Setup MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("wellness_tourism")
    
    # Train models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"f1_score": f1, "accuracy": acc, "roc_auc": auc})
            mlflow.sklearn.log_model(model, "model")
            
            print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}, AUC={auc:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name
    
    print(f"\nBest Model: {best_name} (F1: {best_score:.4f})")
    
    # Save and upload best model
    os.makedirs("tourism_project/model_building", exist_ok=True)
    joblib.dump(best_model, "tourism_project/model_building/best_model.joblib")
    joblib.dump(X_train.columns.tolist(), "tourism_project/model_building/feature_names.joblib")
    
    create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
    
    for file_name in ["best_model.joblib", "label_encoders.joblib", "feature_names.joblib"]:
        file_path = f"tourism_project/model_building/{file_name}"
        if os.path.exists(file_path):
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=MODEL_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
    
    print("✓ Model training and registration completed!")

if __name__ == "__main__":
    main()
