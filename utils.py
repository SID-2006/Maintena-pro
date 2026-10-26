import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "ai4i2020.csv"
SUMMARY_PATH = MODELS_DIR / "summary_metrics.json"

@st.cache_resource(show_spinner=False)
def load_models():
    """Loads all pre-trained models and transformers from disk."""
    if not (MODELS_DIR / "failure_model.pkl").exists():
        raise FileNotFoundError("Models not found. Run train_models.py first.")

    models = {
        "preprocessor": joblib.load(MODELS_DIR / "preprocessor.pkl"),
        "failure_model": joblib.load(MODELS_DIR / "failure_model.pkl"),
        "baseline_models": joblib.load(MODELS_DIR / "baseline_models.pkl"),
        "regression_model": joblib.load(MODELS_DIR / "regression_model.pkl"),
        "cluster_scaler": joblib.load(MODELS_DIR / "cluster_scaler.pkl"),
        "kmeans_model": joblib.load(MODELS_DIR / "kmeans_model.pkl"),
        "gmm_model": joblib.load(MODELS_DIR / "gmm_model.pkl"),
        "pca_model": joblib.load(MODELS_DIR / "pca_model.pkl"),
    }
    return models

@st.cache_data(show_spinner=False)
def load_summary_metrics():
    """Loads precomputed training & evaluation metrics."""
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads and caches the AI4I 2020 predictive maintenance dataset."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return None

def compute_physics_indicators(type_val, air_temp, proc_temp, rot_speed, torque, tool_wear):
    """
    Computes domain physics indicators from the AI4I 2020 dataset specs:
    - Power P = 2 * pi * N * Torque / 60 (Watts)
    - Temp Delta = Process Temp - Air Temp (Kelvin)
    - Overstrain Product = Torque * Tool Wear (Nm * min)
    """
    # Angular velocity omega = 2 * pi * rpm / 60
    power_watts = torque * (rot_speed * (2 * np.pi / 60))
    temp_delta = proc_temp - air_temp
    overstrain_val = torque * tool_wear

    # Threshold limits from AI4I dataset documentation
    thresholds = {
        "L": 11000,
        "M": 12000,
        "H": 13000
    }
    osf_limit = thresholds.get(type_val, 11000)

    # Flag potential failure triggers
    active_risks = []
    
    # Tool Wear Failure (TWF): tool reaches replacement life (usually 200 - 240 mins)
    if tool_wear >= 200:
        active_risks.append({
            "code": "TWF",
            "name": "Tool Wear Failure Risk",
            "reason": f"Tool wear ({tool_wear:.0f} min) is at or above critical threshold (>= 200 min).",
            "severity": "CRITICAL" if tool_wear >= 230 else "WARNING"
        })

    # Heat Dissipation Failure (HDF): delta T < 8.6 K and speed < 1380 rpm
    if temp_delta < 8.6 and rot_speed < 1380:
        active_risks.append({
            "code": "HDF",
            "name": "Heat Dissipation Failure Risk",
            "reason": f"Low temperature delta ({temp_delta:.1f} K < 8.6 K) coupled with low speed ({rot_speed} rpm < 1380 rpm) inhibits thermal dissipation.",
            "severity": "CRITICAL"
        })

    # Power Failure (PWF): Power < 3500 W or > 9000 W
    if power_watts < 3500 or power_watts > 9000:
        active_risks.append({
            "code": "PWF",
            "name": "Power Anomaly Failure Risk",
            "reason": f"Mechanical power ({power_watts:.0f} W) out of safe operating envelope [3,500 W - 9,000 W].",
            "severity": "CRITICAL"
        })

    # Overstrain Failure (OSF): Overstrain product exceeds limit
    if overstrain_val > osf_limit:
        active_risks.append({
            "code": "OSF",
            "name": "Overstrain Failure Risk",
            "reason": f"Torque x Tool Wear load ({overstrain_val:.0f} min·Nm) exceeds Type-{type_val} limit ({osf_limit} min·Nm).",
            "severity": "CRITICAL"
        })

    return {
        "power_watts": power_watts,
        "temp_delta": temp_delta,
        "overstrain_val": overstrain_val,
        "osf_limit": osf_limit,
        "active_risks": active_risks
    }

def predict_single(input_dict, selected_model_name="Tuned Random Forest"):
    """
    Performs full diagnostic inference for single machine sensor readings:
    - Classification: Machine Failure + Failure Probability
    - Regression: Expected Process Temperature
    - Clustering: Operating Regime (KMeans & GMM)
    - Domain Risk Factors & Prescriptive Recommendations
    """
    models = load_models()
    preprocessor = models["preprocessor"]

    # Choose classifier
    if selected_model_name == "Tuned Random Forest":
        clf = models["failure_model"]
    else:
        clf = models["baseline_models"].get(selected_model_name, models["failure_model"])

    # Prepare DataFrame for Preprocessor
    input_df = pd.DataFrame([{
        "Type": input_dict["Type"],
        "Air temperature [K]": float(input_dict["Air temperature [K]"]),
        "Process temperature [K]": float(input_dict["Process temperature [K]"]),
        "Rotational speed [rpm]": float(input_dict["Rotational speed [rpm]"]),
        "Torque [Nm]": float(input_dict["Torque [Nm]"]),
        "Tool wear [min]": float(input_dict["Tool wear [min]"])
    }])

    # Transform
    X_proc = preprocessor.transform(input_df)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    # Predict Failure
    fail_pred = int(clf.predict(X_proc)[0])
    if hasattr(clf, "predict_proba"):
        fail_prob = float(clf.predict_proba(X_proc)[0][1])
    else:
        fail_prob = 1.0 if fail_pred == 1 else 0.0

    # Predict Process Temp via Linear Regression
    reg_features = [
        "Air temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    X_reg = input_df[reg_features]
    reg_model = models["regression_model"]
    pred_process_temp = float(reg_model.predict(X_reg)[0])

    # Cluster Regime
    cluster_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    cluster_data = input_df[cluster_features]
    cluster_scaled = models["cluster_scaler"].transform(cluster_data)
    kmeans_cluster = int(models["kmeans_model"].predict(cluster_scaled)[0])
    gmm_cluster = int(models["gmm_model"].predict(cluster_scaled)[0])
    gmm_probs = models["gmm_model"].predict_proba(cluster_scaled)[0].tolist()

    # Domain Physics
    physics = compute_physics_indicators(
        input_dict["Type"],
        float(input_dict["Air temperature [K]"]),
        float(input_dict["Process temperature [K]"]),
        float(input_dict["Rotational speed [rpm]"]),
        float(input_dict["Torque [Nm]"]),
        float(input_dict["Tool wear [min]"])
    )

    # Regime Descriptions
    regime_names = {
        0: "Regime 0: Heavy-Duty Thermal Load (High Temp, Elevated Torque)",
        1: "Regime 1: Standard Cool Steady-State (Nominal Load & Temps)",
        2: "Regime 2: High-Speed Light-Torque (High RPM, Low Torque Profile)"
    }

    # Prescriptive Maintenance Recommendations
    recommendations = []
    if fail_prob >= 0.5 or len(physics["active_risks"]) > 0:
        if any(r["code"] == "TWF" for r in physics["active_risks"]):
            recommendations.append("Immediate tool bit exchange required: cutter has reached terminal wear limit.")
        if any(r["code"] == "HDF" for r in physics["active_risks"]):
            recommendations.append("Thermal dissipation deficiency detected: inspect coolant circulation pumps and spindle heat sinks.")
        if any(r["code"] == "PWF" for r in physics["active_risks"]):
            recommendations.append("Power anomaly: verify electrical drive motor current draw and gearbox transmission lubrication.")
        if any(r["code"] == "OSF" for r in physics["active_risks"]):
            recommendations.append("Structural overstrain: reduce feed rate or torque setpoints immediately to avoid catastrophic tool fracture.")
        if not recommendations:
            recommendations.append("Machine telemetry indicates an elevated anomaly risk profile. Schedule priority technician diagnostic.")
    else:
        recommendations.append("Equipment is operating well within nominal OEM tolerances. Next routine inspection scheduled per standard protocol.")

    return {
        "failure_prediction": fail_pred,
        "failure_probability": fail_prob,
        "predicted_process_temp": pred_process_temp,
        "actual_process_temp": float(input_dict["Process temperature [K]"]),
        "temp_residual": float(input_dict["Process temperature [K]"]) - pred_process_temp,
        "kmeans_cluster": kmeans_cluster,
        "regime_name": regime_names.get(kmeans_cluster, f"Cluster {kmeans_cluster}"),
        "gmm_cluster": gmm_cluster,
        "gmm_probabilities": gmm_probs,
        "physics": physics,
        "recommendations": recommendations
    }

def process_batch(df_batch, selected_model_name="Tuned Random Forest"):
    """
    Processes a batch of sensor records from an uploaded DataFrame:
    Runs preprocessing, failure prediction, probability, process temp estimation, and clustering.
    """
    models = load_models()
    preprocessor = models["preprocessor"]

    if selected_model_name == "Tuned Random Forest":
        clf = models["failure_model"]
    else:
        clf = models["baseline_models"].get(selected_model_name, models["failure_model"])

    reg_model = models["regression_model"]
    cluster_scaler = models["cluster_scaler"]
    kmeans_model = models["kmeans_model"]

    req_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    missing = [c for c in req_cols if c not in df_batch.columns]
    if missing:
        raise ValueError(f"Batch file is missing required columns: {missing}")

    X_in = df_batch[req_cols].copy()
    X_proc = preprocessor.transform(X_in)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    preds = clf.predict(X_proc)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_proc)[:, 1]
    else:
        probs = preds.astype(float)

    reg_cols = ["Air temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    pred_temps = reg_model.predict(X_in[reg_cols])

    cluster_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    cluster_scaled = cluster_scaler.transform(X_in[cluster_cols])
    clusters = kmeans_model.predict(cluster_scaled)

    # Power & Physics
    powers = X_in["Torque [Nm]"] * (X_in["Rotational speed [rpm]"] * (2 * np.pi / 60))
    deltas = X_in["Process temperature [K]"] - X_in["Air temperature [K]"]

    result_df = df_batch.copy()
    result_df["Failure_Risk_Pred"] = preds
    result_df["Failure_Probability"] = np.round(probs, 4)
    result_df["Risk_Level"] = np.where(probs >= 0.65, "CRITICAL", np.where(probs >= 0.35, "WARNING", "NORMAL"))
    result_df["Est_Process_Temp_K"] = np.round(pred_temps, 2)
    result_df["Temp_Delta_K"] = np.round(deltas, 2)
    result_df["Power_Watts"] = np.round(powers, 1)
    result_df["Operating_Cluster"] = clusters

    return result_df
