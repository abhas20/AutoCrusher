import pandas as pd
from fastapi import HTTPException
import app.state as state

def predict_fair_outcome(features: dict, model_filename: str = "fair_adult_model.pkl"):
    """
    Business logic for processing input features, aligning dummy columns,
    scaling numerical fields, and serving fair predictions.
    Uses the cache-validated model lookup.
    """
    # Fetch/verify model bundle from the cache validation lookup
    bundle = state.load_model(model_filename)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_filename}' is not loaded or does not exist. Please run mitigation first."
        )
            
    try:
        # Convert input dictionary to DataFrame
        df_in = pd.DataFrame([features])
        
        # Drop target label if passed in features by mistake
        label_col = bundle.get("label_col")
        if label_col in df_in.columns:
            df_in = df_in.drop(columns=[label_col])
            
        # Drop features removed during mitigation
        removed_cols = bundle.get("removed_cols", [])
        df_in = df_in.drop(columns=removed_cols, errors="ignore")
        
        # Perform one-hot encoding for categorical variables
        cat_cols = df_in.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df_in = pd.get_dummies(df_in, columns=cat_cols, drop_first=True)
            
        # Align with the trained model's features (fill missing dummies with 0)
        feature_cols = bundle.get("feature_columns", [])
        for col in feature_cols:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[feature_cols]
        
        # Transform inputs using the fitted StandardScaler
        scaler = bundle.get("scaler")
        X_scaled = scaler.transform(df_in)
        
        # 7. Generate fair predictions
        model = bundle.get("model")
        try:
            prediction = int(model.predict(X_scaled, random_state=42)[0])
        except TypeError:
            # Fallback for standard classifiers that don't accept random_state in predict()
            prediction = int(model.predict(X_scaled)[0])
        
        return {
            "prediction": prediction,
            "label_meaning": "Positive Class (e.g. >50K / Approved)" if prediction == 1 else "Negative Class (e.g. <=50K / Rejected)",
            "model_used": model_filename,
            "primary_sensitive_feature": bundle.get("primary_sensitive_col"),
            "mitigated_features": bundle.get("removed_cols")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
