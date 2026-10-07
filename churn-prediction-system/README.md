# Customer Churn Prediction System
Production-ready Machine Learning system predicting customer churn using XGBoost and served via FastAPI.


**Note:** src/make_sample_data.py was added (previously no data/dataset.csv existed) to generate a synthetic dataset with a simple learnable churn signal, so train.py is runnable out of the box. Verified: the preprocessing pipeline was run end to end against this generated data successfully. Replace data/dataset.csv with a real dataset (e.g. the Telco Customer Churn dataset) for anything beyond a pipeline smoke test.
