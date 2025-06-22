import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, balanced_accuracy_score
)
import argparse
import joblib
import os

# Set MLflow tracking
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Student Performance CI")

# Load data
data = pd.read_csv("Student_performance_processed_data.csv")
X = data.drop("GradeClass", axis=1)
y = data["GradeClass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop("GradeClass", axis=1),
#     data["GradeClass"],
#     random_state=42,
#     test_size=0.2
# )

input_example = X_train[0:5]
dataset = mlflow.data.from_pandas(data)

mlflow.autolog()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

accuracy = model.score(X_test, y_test)

mlflow.log_input(dataset, context="training")

os.makedirs("artefak_model", exist_ok=True)
joblib.dump(model, "artefak_model/model.pkl")