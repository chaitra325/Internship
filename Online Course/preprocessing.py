# inference.py
import pickle
import numpy as np
import pandas as pd

# Load artifacts saved from Jupyter
with open("course_success_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

scaler = artifacts["scaler"]
ohe = artifacts["ohe"]
numeric_features = artifacts["numeric_features"]
categorical_features = artifacts["categorical_features"]
model = artifacts["model"]

def rebuild_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    
    #Recreate engineered features exactly as in the notebook
    

    df = df_raw.copy()

    # From notebook: composite features
    df["log_reviews"] = np.log1p(df["reviews"])
    df["success_score"] = 0.6 * df["rating"] + 0.4 * df["log_reviews"]

    # Instructor popularity
    df["instr_log_total_reviews"] = np.log1p(df["instr_total_reviews"])

    # Buckets (same bins and labels as notebook)
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[-0.01, 0, 1000, 3000, np.inf],
        labels=["free", "low", "medium", "high"]
    )

    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[-0.01, 2, 10, 30, np.inf],
        labels=["very_short", "short", "medium", "long"]
    )

    return df

def preprocess_for_model(raw_dict: dict) -> np.ndarray:
   
    #Take raw input dict, build final feature matrix for the model.
 
    df_raw = pd.DataFrame([raw_dict])
    df_feat = rebuild_engineered_features(df_raw)

    X_num = df_feat[numeric_features]
    X_cat = df_feat[categorical_features]

    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)

    X_final = np.hstack([X_num_scaled, X_cat_encoded])
    return X_final

def predict_success(raw_dict: dict):
    #Given raw input dict, return predicted label and probability.
    #Returns (label, probability_of_high_success)
    
    X = preprocess_for_model(raw_dict)
    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)
    return label, float(proba)
