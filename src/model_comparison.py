import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from data_prep import preprocess_data


def load_and_split(data_path="data/vehicle_maintenance_data.csv"):
    """Load CSV, preprocess, and return train/test splits."""
    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def tune_random_forest(X_train, y_train):
    """Use GridSearchCV to find the best n_estimators for RandomForest."""
    param_grid = {"n_estimators": [50, 100, 150, 200, 250]}
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("\n=== GridSearchCV Results ===")
    print(f"Best n_estimators : {grid.best_params_['n_estimators']}")
    print(f"Best CV Accuracy  : {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_logistic_regression(X_train, y_train):
    """Train a LogisticRegression model with feature scaling."""
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    return lr_pipeline


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy & classification report; return accuracy and predictions."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))
    return acc, y_pred


def plot_confusion_matrices(y_test, rf_pred, lr_pred, save_path="models/confusion_matrix_comparison.png"):
    """Plot side-by-side confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Random Forest (Tuned)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Oranges", ax=axes[1])
    axes[1].set_title("Logistic Regression")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix plot saved to {save_path}")


def main():
    # ---- Load data ----
    X_train, X_test, y_train, y_test = load_and_split()

    # ---- Train models ----
    print("\n>>> Tuning Random Forest with GridSearchCV ...")
    rf_model = tune_random_forest(X_train, y_train)

    print("\n>>> Training Logistic Regression ...")
    lr_model = train_logistic_regression(X_train, y_train)

    # ---- Evaluate ----
    rf_acc, rf_pred = evaluate_model("Random Forest (Tuned)", rf_model, X_test, y_test)
    lr_acc, lr_pred = evaluate_model("Logistic Regression", lr_model, X_test, y_test)

    # ---- Confusion matrices ----
    plot_confusion_matrices(y_test, rf_pred, lr_pred)

    # ---- Save best model ----
    os.makedirs("models", exist_ok=True)
    if rf_acc >= lr_acc:
        best_name = "Random Forest (Tuned)"
        joblib.dump(rf_model, "models/vehicle_model.pkl")
    else:
        best_name = "Logistic Regression"
        joblib.dump(lr_model, "models/vehicle_model.pkl")

    print(f"\n✅ Best model: {best_name}")
    print(f"   RF Accuracy : {rf_acc:.4f}")
    print(f"   LR Accuracy : {lr_acc:.4f}")
    print(f"   Saved to models/vehicle_model.pkl")


if __name__ == "__main__":
    main()
