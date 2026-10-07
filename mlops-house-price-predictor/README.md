# MLOps House Price Predictor
End-to-End ML pipeline with Experiment Tracking via MLflow and REST API serving via FastAPI.


**Note:** app.py previously returned a hardcoded mock formula instead of using the trained model at all - fixed so it now loads and predicts from the model saved by src/train_mlflow.py (run that first, then start the API).
