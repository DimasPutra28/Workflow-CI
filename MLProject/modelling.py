import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os

if __name__ == "__main__":

    np.random.seed(42)

    # get file path from arguments
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "croprecommendation_preprocessing.csv")
    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("GradeClass", axis=1),
        data["GradeClass"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]
    dataset = mlflow.data.from_pandas(data)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

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