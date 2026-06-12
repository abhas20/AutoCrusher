import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
from pathlib import Path

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.preprocessing import CorrelationRemover

# ── Constants and Thresholds ──────────────────────────────────────────────────
BIAS_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "equalized_odds_difference":     0.10,
    "disparate_impact_ratio":        0.80,   # DIR below 0.80 ->considered a strong bias signal
}

CANDIDATE_MODELS = {
    "LogisticRegression":     LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree":           DecisionTreeClassifier(random_state=42),
    "RandomForest":           RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting":       GradientBoostingClassifier(random_state=42),
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def _safe_bin(series: pd.Series, q: int = 4) -> pd.Series:
    """Bin continuous numeric columns into quartile buckets."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        try:
            return pd.qcut(series, q=q, duplicates="drop").astype(str)
        except Exception:
            median = series.median()
            return (series > median).map({True: "high", False: "low"})
    return series.astype(str)


def _cramers_v(col: pd.Series, target: pd.Series) -> float:
    """
    Cramér's V — symmetric measure of association between two
    categorical variables. Range [0, 1]; higher = stronger association.
    """
    contingency = pd.crosstab(col, target)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    r, k = contingency.shape
    if n <= 1:
        return 0.0
    # Bias-corrected formula
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _mutual_information(col: pd.Series, target: pd.Series) -> float:
    """
    Normalized Mutual Information between a feature and the label.
    Captures non-linear dependencies. Range [0, 1].
    """
    col_codes = col.astype("category").cat.codes
    tgt_codes = target.astype("category").cat.codes
    mi = mutual_info_score(col_codes, tgt_codes)
    # Normalize by mean entropy
    h_col = mutual_info_score(col_codes, col_codes)
    h_tgt = mutual_info_score(tgt_codes, tgt_codes)
    denom = (h_col + h_tgt) / 2
    return float(mi / denom) if denom > 0 else 0.0


def _outcome_imbalance(col: pd.Series, target: pd.Series) -> float:
    """Measures outcome disparity across groups."""
    df = pd.DataFrame({"col": col, "target": target})
    group_rates = df.groupby("col")["target"].mean()
    if len(group_rates) < 2:
        return 0.0
    return float(group_rates.std() / (group_rates.mean() + 1e-9))


def _ensure_binary_numeric(series: pd.Series, col_name: str) -> tuple[pd.Series, dict]:
    """Ensures a series is binary and encodes it as 0/1 integers."""
    unique_vals = series.dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Column '{col_name}' must be binary (exactly 2 unique values), but has {len(unique_vals)}: {unique_vals}")
    
    if pd.api.types.is_numeric_dtype(series) and set(unique_vals).issubset({0, 1}):
        return series.astype(int), {str(v): int(v) for v in unique_vals}
    
    mapping = {str(val): idx for idx, val in enumerate(sorted(unique_vals))}
    encoded = series.map(lambda x: mapping.get(str(x)) if pd.notna(x) else np.nan)
    return encoded.astype(int), mapping


def _reconstruct_from_onehot(df: pd.DataFrame, onehot_cols: list) -> pd.Series:
    """Converts one-hot encoded columns back into a single categorical Series."""
    missing = [c for c in onehot_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    
    subset = df[onehot_cols]
    
    def _row_to_label(row):
        active = row[row == 1].index.tolist()
        if active:
            return active[0]
        return "unknown"
    
    return subset.apply(_row_to_label, axis=1)


def _binarize_continuous(series: pd.Series, strategy: str = "median", threshold: float = None) -> tuple[pd.Series, float]:
    """Converts continuous predictions/scores to binary 0/1 for metric evaluation."""
    if strategy == "threshold" and threshold is not None:
        cutoff = threshold
    elif strategy == "mean":
        cutoff = series.mean()
    else:
        cutoff = series.median()
        
    binary = (series >= cutoff).astype(int)
    return binary, round(float(cutoff), 4)


def _compute_bias_metrics(y_true: pd.Series, y_pred: pd.Series, sensitive_series: pd.Series) -> dict:
    """Computes demographic parity difference, disparate impact, and equalized odds."""
    # Bin if sensitive_series is continuous
    binned_sensitive = _safe_bin(sensitive_series)
    
    dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=binned_sensitive))
    dir_ = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=binned_sensitive))
    eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=binned_sensitive))
    
    verdict = "FAIR"
    reasons = []
    if abs(dpd) > BIAS_THRESHOLDS["demographic_parity_difference"]:
        verdict = "BIASED"
        reasons.append(f"DPD={dpd:.3f} > {BIAS_THRESHOLDS['demographic_parity_difference']}")
    if dir_ < BIAS_THRESHOLDS["disparate_impact_ratio"]:
        verdict = "BIASED"
        reasons.append(f"DIR={dir_:.3f} < {BIAS_THRESHOLDS['disparate_impact_ratio']}")
    if abs(eod) > BIAS_THRESHOLDS["equalized_odds_difference"]:
        verdict = "BIASED"
        reasons.append(f"EOD={eod:.3f} > {BIAS_THRESHOLDS['equalized_odds_difference']}")
        
    return {
        "demographic_parity_difference": round(dpd, 4),
        "disparate_impact_ratio": round(dir_, 4),
        "equalized_odds_difference": round(eod, 4),
        "verdict": verdict,
        "reasons": reasons
    }


def _load_and_prepare(dataset_path: str, label_col: str, drop_cols: list = None, pred_col: str = None):
    """Loads dataset, encodes label to 0/1, encodes categoricals, and handles missing values."""
    df_raw = pd.read_csv(dataset_path)
    df = df_raw.copy()
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    if pred_col and pred_col in df.columns:
        df = df.drop(columns=[pred_col], errors="ignore")
        
    # Encode label to 0/1
    y_raw = df[label_col].copy()
    y, _ = _ensure_binary_numeric(y_raw, label_col)
    
    X = df.drop(columns=[label_col])
    
    # Exclude prediction columns if they somehow exist in the features
    for col in ["y_pred", "model_predictions", "predicted", "score_factor"]:
        if col in X.columns:
            X = X.drop(columns=[col])
            
    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, df_raw

# ── Primary Tools ─────────────────────────────────────────────────────────

def analyze_dataset_fields(dataset_path: str, label_col: str, pred_col: str = None) -> str:
    """
    Scans a dataset's columns to identify:
      1. Useless or unnecessary fields (constant columns, unique identifiers, high missingness).
      2. Sensitive proxy candidates (demographic keywords or high statistical association with the label).
    """
    try:
        df = pd.read_csv(dataset_path)
        if label_col not in df.columns:
            return json.dumps({"status": "error", "message": f"Label column '{label_col}' not found in dataset."})
            
        total_rows = len(df)
        useless_fields = {}
        proxy_candidates = {}
        
        DEMOGRAPHIC_KEYWORDS = {
            "gender", "sex", "race", "ethnicity", "age", "religion", "caste",
            "nationality", "marital", "disability", "income", "zip", "postal",
            "tribe", "color", "colour", "origin", "citizen",
        }
        
        SENSITIVITY_SCORE_THRESHOLD = 0.30
        target = df[label_col].copy()
        
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 2:
            target_binned, _ = _binarize_continuous(target)
        else:
            target_binned, _ = _ensure_binary_numeric(target, label_col)

        for col in df.columns:
            if col == label_col or (pred_col and col == pred_col):
                continue
                
            series = df[col]
            num_unique = series.nunique()
            null_ratio = series.isnull().mean()
            col_lower = col.lower()
            
            reasons = []
            
            # Check for useless/unnecessary fields
            if num_unique <= 1:
                reasons.append("Constant column (only 1 unique value)")
            if null_ratio > 0.80:
                reasons.append(f"High missingness ({null_ratio:.1%} null values)")
                
            id_keywords = {"id", "uuid", "uid", "index", "key", "serial", "number"}
            if (num_unique == total_rows) and (series.dtype == object or pd.api.types.is_integer_dtype(series)):
                reasons.append("Unique row identifier (unique values match total row count)")
            elif any(kw in col_lower for kw in id_keywords) and (num_unique > 0.90 * total_rows):
                reasons.append(f"Likely ID/key column ({num_unique} unique values for {total_rows} rows)")
                
            if reasons:
                useless_fields[col] = reasons
                continue
                
            # Check for sensitive proxy candidates
            proxy_reasons = []
            matched_kw = [kw for kw in DEMOGRAPHIC_KEYWORDS if kw in col_lower]
            if matched_kw:
                proxy_reasons.append(f"Name matches demographic keyword(s): {matched_kw}")
            if series.dtype == object and 2 <= num_unique <= 10:
                proxy_reasons.append(f"Low-cardinality categorical ({num_unique} unique values)")
            if pd.api.types.is_numeric_dtype(series) and num_unique == 2:
                proxy_reasons.append("Binary numeric (possible encoded category)")
                
            # Statistical check
            binned_col = _safe_bin(series)
            cv = _cramers_v(binned_col, target_binned)
            nmi = _mutual_information(binned_col, target_binned)
            oi = _outcome_imbalance(binned_col, target_binned)
            
            composite = 0.35 * cv + 0.45 * nmi + 0.20 * oi
            if composite >= SENSITIVITY_SCORE_THRESHOLD:
                proxy_reasons.append(
                    f"Strong statistical proxy signal (composite={composite:.3f} >= threshold {SENSITIVITY_SCORE_THRESHOLD}). "
                    f"Cramér's V={cv:.3f}, NMI={nmi:.3f}, Outcome Imbalance={oi:.3f}"
                )
                
            if proxy_reasons:
                proxy_candidates[col] = {
                    "reasons": proxy_reasons,
                    "metrics": {
                        "composite_score": round(float(composite), 4),
                        "cramers_v": round(float(cv), 4),
                        "mutual_information": round(float(nmi), 4),
                        "outcome_imbalance": round(float(oi), 4)
                    }
                }
                
        return json.dumps({
            "status": "ok",
            "useless_fields": useless_fields,
            "sensitive_proxy_candidates": proxy_candidates,
            "summary": f"Identified {len(useless_fields)} useless fields and {len(proxy_candidates)} sensitive proxy candidates."
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def calculate_fairlearn_bias(
    dataset_path: str,
    label_col: str,
    pred_col: str,
    sensitive_feature_name,  # str or list of str (one-hot group)
    binarize_strategy: str = "median",
    binarize_threshold: float = None,
) -> str:
    """Calculates Fairlearn bias metrics for a sensitive feature."""
    try:
        df = pd.read_csv(dataset_path)
        
        for col in [label_col, pred_col]:
            if col not in df.columns:
                return json.dumps({"status": "error", "message": f"Column '{col}' not found in dataset."})
                
        is_onehot = isinstance(sensitive_feature_name, list)
        
        if is_onehot:
            feature_data = _reconstruct_from_onehot(df, sensitive_feature_name)
            feature_label = f"[{', '.join(sensitive_feature_name)}] (reconstructed)"
        else:
            if sensitive_feature_name not in df.columns:
                return json.dumps({"status": "error", "message": f"Column '{sensitive_feature_name}' not found."})
            feature_data = _safe_bin(df[sensitive_feature_name])
            feature_label = sensitive_feature_name
            
        y_true_raw = df[label_col]
        y_pred_raw = df[pred_col]
        
        # Binarize if continuous
        if pd.api.types.is_numeric_dtype(y_true_raw) and y_true_raw.nunique() > 2:
            y_true, true_cutoff = _binarize_continuous(y_true_raw, binarize_strategy, binarize_threshold)
        else:
            y_true, _ = _ensure_binary_numeric(y_true_raw, label_col)
            
        if pd.api.types.is_numeric_dtype(y_pred_raw) and y_pred_raw.nunique() > 2:
            y_pred, pred_cutoff = _binarize_continuous(y_pred_raw, binarize_strategy, binarize_threshold)
        else:
            y_pred, _ = _ensure_binary_numeric(y_pred_raw, pred_col)
            
        metrics = _compute_bias_metrics(y_true, y_pred, feature_data)
        
        return json.dumps({
            "status": "ok",
            "feature_tested": feature_label,
            "was_onehot_reconstructed": is_onehot,
            "groups_found": feature_data.unique().tolist(),
            "demographic_parity_difference": metrics["demographic_parity_difference"],
            "disparate_impact_ratio": metrics["disparate_impact_ratio"],
            "equalized_odds_difference": metrics["equalized_odds_difference"],
            "verdict": metrics["verdict"],
            "failed_reasons": metrics["reasons"],
            "plain_english": (
                f"Across groups in '{feature_label}': "
                f"demographic parity gap = {abs(metrics['demographic_parity_difference']):.1%} "
                f"({'exceeds' if 'DPD' in ''.join(metrics['reasons']) else 'within'} ±10%), "
                f"disparate impact ratio = {metrics['disparate_impact_ratio']:.2f} "
                f"({'below' if 'DIR' in ''.join(metrics['reasons']) else 'above'} 0.80), "
                f"equalized odds gap = {abs(metrics['equalized_odds_difference']):.1%} "
                f"({'exceeds' if 'EOD' in ''.join(metrics['reasons']) else 'within'} ±10%)."
            )
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def apply_bias_mitigation(
    dataset_path: str,
    label_col: str,
    biased_columns: str,          # JSON list: '["col_a", "col_b"]'
    primary_sensitive_col: str,   
    pred_col: str = None,
    output_filename: str = "fair_model.pkl",
) -> str:
    """Applies mitigation (CorrelationRemover and ExponentiatedGradient) and retrains candidate models."""
    try:
        biased_cols = json.loads(biased_columns)
        
        # Load and prepare dataset
        X, y, df_raw = _load_and_prepare(dataset_path, label_col, pred_col=pred_col)
        
        # Determine demographic parity difference on primary sensitive column
        sensitive_series = df_raw[primary_sensitive_col].reset_index(drop=True)
        sensitive_binned = _safe_bin(sensitive_series)
        
        # Remove other biased columns except the primary sensitive column
        cols_to_remove = [c for c in biased_cols if c != primary_sensitive_col]
        
        # Load and prepare again dropping cols to remove and pred_col if exists
        X, y, df_raw = _load_and_prepare(dataset_path, label_col, drop_cols=cols_to_remove, pred_col=pred_col)
        
        # Align sensitive feature
        sensitive_series = df_raw[primary_sensitive_col].reset_index(drop=True)
        sensitive_binned = _safe_bin(sensitive_series)
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Train-test split
        X_tr, X_te, y_tr, y_te, sf_tr, sf_te = train_test_split(
            X, y, sensitive_binned,
            test_size=0.25, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # In-processing constraint (DemographicParity as default)
        fairness_constraint = DemographicParity()
        
        model_results = {}
        best_model = None
        best_score = -1
        best_name = ""
        
        for model_name, base_model in CANDIDATE_MODELS.items():
            try:
                fair_model = ExponentiatedGradient(
                    estimator=base_model,
                    constraints=fairness_constraint,
                    max_iter=30
                )
                fair_model.fit(X_tr_s, y_tr, sensitive_features=sf_tr)
                y_pred = fair_model.predict(X_te_s)
                
                acc = round(accuracy_score(y_te, y_pred), 4)
                f1 = round(f1_score(y_te, y_pred, zero_division=0), 4)
                
                # Fairness evaluation
                bias = _compute_bias_metrics(y_te, pd.Series(y_pred), sf_te.reset_index(drop=True))
                
                # Composite score: 60% accuracy + 40% fairness
                dpd_viol = max(0, abs(bias["demographic_parity_difference"]) - 0.10)
                dir_viol = max(0, 0.80 - bias["disparate_impact_ratio"])
                eod_viol = max(0, abs(bias["equalized_odds_difference"]) - 0.10)
                fairness_score = max(0, 1 - (dpd_viol + dir_viol + eod_viol))
                
                composite = round(0.60 * acc + 0.40 * fairness_score, 4)
                
                model_results[model_name] = {
                    "accuracy": acc,
                    "f1_score": f1,
                    "bias_after": bias,
                    "fairness_score": round(fairness_score, 4),
                    "composite_score": composite
                }
                
                if composite > best_score:
                    best_score = composite
                    best_model = fair_model
                    best_name = model_name
                    
            except Exception as model_err:
                model_results[model_name] = {
                    "status": "failed",
                    "error": str(model_err)
                }
                
        # Save model bundle to output folder
        output_dir = Path("./bias_pipeline_output")
        output_dir.mkdir(exist_ok=True)
        final_path = output_dir / output_filename
        
        with open(final_path, "wb") as f:
            pickle.dump({
                "model": best_model,
                "scaler": scaler,
                "feature_columns": list(X.columns),
                "label_col": label_col,
                "removed_cols": cols_to_remove,
                "primary_sensitive_col": primary_sensitive_col,
                "model_results": model_results,
                "best_model_name": best_name
            }, f)
            
        return json.dumps({
            "status": "ok",
            "best_model_name": best_name,
            "best_composite_score": best_score,
            "final_pkl_path": str(final_path),
            "columns_removed": cols_to_remove,
            "model_results": model_results,
            "summary": f"Successfully trained {best_name} with composite score {best_score:.4f} and exported model to {final_path}."
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error running bias mitigation: {str(e)}"})


def generate_baseline_predictions(dataset_path: str, label_col: str) -> tuple[str, str]:
    """
    Trains a quick baseline model (LogisticRegression) on the dataset
    and saves the predictions as a new column in a temporary CSV file.
    
    Returns:
        (new_csv_path, prediction_column_name)
    """
    # Load and prepare features
    X, y, df_raw = _load_and_prepare(dataset_path, label_col)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    
    # Generate predictions on the entire dataset
    df_raw["_baseline_predictions"] = clf.predict(X_scaled)
    
    # Save to a new CSV file in the same directory
    orig_path = Path(dataset_path)
    new_path = orig_path.parent / f"{orig_path.stem}_with_preds.csv"
    df_raw.to_csv(new_path, index=False)
    
    print(f"🤖 Automatically generated baseline predictions column '_baseline_predictions' and saved to: {new_path}")
    return str(new_path), "_baseline_predictions"

