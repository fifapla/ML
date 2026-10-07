import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_model.joblib")


def run_experiment():
    mlflow.set_experiment("House_Price_Prediction")

    # NOTE: synthetic data for demo purposes only - replace with a real
    # housing dataset (e.g. the Ames or California housing data) for
    # anything beyond a pipeline smoke test.
    rng = np.random.RandomState(42)
    X = rng.rand(200, 5)
    y = X[:, 0] * 300 + X[:, 1] * 150 + rng.randn(200) * 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        params = {"n_estimators": 50, "learning_rate": 0.05, "random_state": 42}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        # Also save a plain joblib copy so the FastAPI app can load it
        # without depending on an MLflow tracking server being reachable.
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Logged run to MLflow. RMSE: {rmse:.2f}, R2: {r2:.2f}")
        print(f"Saved model for serving at: {MODEL_PATH}")


if __name__ == "__main__":
    run_experiment()
