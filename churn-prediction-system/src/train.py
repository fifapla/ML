import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import load_and_preprocess_data

def train_pipeline():
    data_path = 'data/dataset.csv'
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')
    joblib.dump(model, 'model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

if __name__ == '__main__':
    train_pipeline()
