import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

RANDOM_STATE = 42

def train_and_save():
    data_dir = Path("data")
    csv_path = data_dir / "ai4i2020.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # 1. Regression: Process Temperature [K]
    reg_features = [
        "Air temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    X_reg = df[reg_features]
    y_reg = df["Process temperature [K]"]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=RANDOM_STATE
    )

    reg_model = LinearRegression()
    reg_model.fit(X_reg_train, y_reg_train)
    y_reg_pred = reg_model.predict(X_reg_test)

    reg_metrics = {
        "MAE": float(mean_absolute_error(y_reg_test, y_reg_pred)),
        "MSE": float(mean_squared_error(y_reg_test, y_reg_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
        "R2": float(r2_score(y_reg_test, y_reg_pred)),
        "coefficients": {feat: float(coef) for feat, coef in zip(reg_features, reg_model.coef_)},
        "intercept": float(reg_model.intercept_)
    }
    print("Regression metrics:", reg_metrics)

    # 2. Classification: Machine Failure
    target = "Machine failure"
    excluded_columns = [
        "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", target
    ]
    X = df.drop(columns=[col for col in excluded_columns if col in df.columns])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Train classification models
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Bagging Classifier": BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    }

    classification_comparison = []
    trained_baseline_models = {}

    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        pred = model.predict(X_test_processed)
        trained_baseline_models[name] = model

        classification_comparison.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1 Score": float(f1_score(y_test, pred, zero_division=0)),
            "Confusion Matrix": confusion_matrix(y_test, pred).tolist()
        })

    # Tuned Random Forest (Grid Search as in Notebook)
    print("Running Grid Search for Random Forest...")
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_processed, y_train)
    best_rf_model = grid_search.best_estimator_
    best_pred = best_rf_model.predict(X_test_processed)

    tuned_rf_metrics = {
        "Model": "Tuned Random Forest",
        "Best Parameters": grid_search.best_params_,
        "Accuracy": float(accuracy_score(y_test, best_pred)),
        "Precision": float(precision_score(y_test, best_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, best_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, best_pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_test, best_pred).tolist()
    }
    classification_comparison.append(tuned_rf_metrics)

    # Feature Importance
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    feature_importances = [
        {"Feature": feat.replace("numeric__", "").replace("categorical__", ""), "Importance": float(imp)}
        for feat, imp in zip(raw_feature_names, best_rf_model.feature_importances_)
    ]
    feature_importances = sorted(feature_importances, key=lambda x: x["Importance"], reverse=True)

    # Cross validation scores
    print("Computing Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    kfold_scores = cross_val_score(cv_model, X_train_processed, y_train, cv=kf, scoring="f1", n_jobs=-1).tolist()
    stratified_scores = cross_val_score(cv_model, X_train_processed, y_train, cv=skf, scoring="f1", n_jobs=-1).tolist()

    # 3. Clustering: Operating Regime
    cluster_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    cluster_data = df[cluster_features]
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_data)

    print("Fitting KMeans & GMM...")
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(cluster_scaled)

    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(cluster_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    cluster_2d = pca.fit_transform(cluster_scaled)

    cluster_df = df.copy()
    cluster_df["Cluster"] = kmeans_labels
    cluster_summary = cluster_df.groupby("Cluster")[cluster_features].mean().round(2).to_dict()

    silhouette_km = float(silhouette_score(cluster_scaled, kmeans_labels))
    silhouette_gm = float(silhouette_score(cluster_scaled, gmm_labels))

    # Save Models
    print("Saving models to disk...")
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    joblib.dump(best_rf_model, model_dir / "failure_model.pkl")
    joblib.dump(trained_baseline_models, model_dir / "baseline_models.pkl")
    joblib.dump(reg_model, model_dir / "regression_model.pkl")
    joblib.dump(cluster_scaler, model_dir / "cluster_scaler.pkl")
    joblib.dump(kmeans, model_dir / "kmeans_model.pkl")
    joblib.dump(gmm, model_dir / "gmm_model.pkl")
    joblib.dump(pca, model_dir / "pca_model.pkl")

    # Save summary metadata
    summary_data = {
        "reg_metrics": reg_metrics,
        "classification_comparison": classification_comparison,
        "feature_importances": feature_importances,
        "cv_results": {
            "kfold_f1_scores": kfold_scores,
            "kfold_mean_f1": float(np.mean(kfold_scores)),
            "kfold_std_f1": float(np.std(kfold_scores)),
            "stratified_f1_scores": stratified_scores,
            "stratified_mean_f1": float(np.mean(stratified_scores))
        },
        "clustering_metrics": {
            "silhouette_kmeans_k3": silhouette_km,
            "silhouette_gmm_k3": silhouette_gm,
            "cluster_means": cluster_summary,
            "cluster_sizes": [int(s) for s in np.bincount(kmeans_labels)]
        },
        "dataset_stats": {
            "total_records": len(df),
            "failure_count": int(df["Machine failure"].sum()),
            "normal_count": int(len(df) - df["Machine failure"].sum()),
            "failure_rate": float(df["Machine failure"].mean()),
            "failure_types": {
                "TWF (Tool Wear Failure)": int(df["TWF"].sum()),
                "HDF (Heat Dissipation Failure)": int(df["HDF"].sum()),
                "PWF (Power Failure)": int(df["PWF"].sum()),
                "OSF (Overstrain Failure)": int(df["OSF"].sum()),
                "RNF (Random Failure)": int(df["RNF"].sum())
            }
        }
    }

    with open(model_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("All models and summaries saved successfully!")

if __name__ == "__main__":
    train_and_save()
