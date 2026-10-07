import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    categorical_cols = ['gender', 'contract_type', 'payment_method']
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
