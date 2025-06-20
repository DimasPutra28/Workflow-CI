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

# Ambil data_path dari argumen
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='Student_performance_processed_data.csv')
args = parser.parse_args()

data = pd.read_csv(args.data_path)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Student Performance")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("GradeClass", axis=1),
    data["GradeClass"],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]
dataset = mlflow.data.from_pandas(data)

with mlflow.start_run():

    mlflow.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example
        )

    accuracy = model.score(X_test, y_test)

    # Log dataset
    mlflow.log_input(dataset, context="training")