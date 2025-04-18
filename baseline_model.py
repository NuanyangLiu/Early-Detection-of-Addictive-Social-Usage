import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)


def convert_watch_time(df: pd.DataFrame, time_col="Watch Time") -> pd.DataFrame:
    """Parse 12h strings → 24h and bucket into periods."""
    dt = pd.to_datetime(
        df[time_col].astype(str).str.strip(), format="%I:%M %p"
    )
    df["Watch Period"] = dt.dt.hour.map(
        lambda h: "morning"   if 5 <= h < 12 else
                  "afternoon" if 12 <= h < 17 else
                  "evening"   if 17 <= h < 21 else
                  "night"
    )
    df.drop(columns=[time_col], inplace=True)
    return df


def feature_prune(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, hardware fields, and leakage columns."""
    drop_cols = [
        "UserID", "Video ID", "DeviceType", "OS",
        "Self Control", "ProductivityLoss", "Satisfaction", "Frequency"
    ]
    return df.drop(columns=drop_cols)


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Re‑bin the 0–7 scale into low/moderate/high."""
    def _band(x):
        if x <= 1:      return "low"
        elif x <= 4:    return "moderate"
        else:           return "high"

    df["AddictionBand"] = df["AddictionLevel"].map(_band)
    return df.drop(columns=["AddictionLevel"])


def plot_productivity_loss(df: pd.DataFrame, outpath="plots/prod_loss.png"):
    """Bar chart of avg. productivity loss by Watch Period."""
    order = ["morning", "afternoon", "evening", "night"]
    mean_loss = (
        df.groupby("Watch Period")["ProductivityLoss"]
          .mean()
          .reindex(order)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(mean_loss.index, mean_loss.values,
            color=["lightblue", "lightgrey", "lightpink", "lightgreen"])
    plt.title("Average Productivity Loss by Time of Day")
    plt.xlabel("Watch Period")
    plt.ylabel("Avg. Productivity Loss")
    plt.ylim(0, mean_loss.max() + 1)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def build_pipeline(cat_cols, num_cols, **rf_kwargs) -> Pipeline:
    """Create a preprocessing + RF pipeline."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    rf = RandomForestClassifier(**rf_kwargs)
    return Pipeline([("pre", pre), ("rf", rf)])


def evaluate_classification(pipe: Pipeline, X_test, y_test):
    """Print class report and return predictions."""
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    return y_pred


def plot_confusion(cm, labels, outpath="plots/confusion.png"):
    """Blues heatmap confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.colorbar()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_roc(pipe: Pipeline, X_test, y_test, labels, outpath="plots/roc.png"):
    """One‑vs‑rest ROC curves with AUC."""
    y_proba = pipe.predict_proba(X_test)
    y_bin = pd.get_dummies(y_test)[labels].values  # ensures correct order

    plt.figure(figsize=(6, 5))
    for i, lab in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Multi‑Class")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    # 1) Load & preprocess
    df = load_data("data/social_media_dataset.csv")
    df = convert_watch_time(df)
    plot_productivity_loss(df)
    df = feature_prune(df)
    df = create_labels(df)

    # 2) Label distribution
    dist = df["AddictionBand"].value_counts(normalize=True) * 100
    print("Overall AddictionBand distribution (%):")
    print(dist.to_frame("percent").round(1), "\n")

    # 3) Prepare features/target
    X = df.drop(columns=["AddictionBand"])
    y = df["AddictionBand"]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # 4) Build & train baseline RF
    pipe = build_pipeline(cat_cols, num_cols, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)

    # 5) Print classification report
    y_pred = evaluate_classification(pipe, X_test, y_test)

    # 6) Compute and report Macro AUROC
    #    a) binarize the true labels
    y_bin = pd.get_dummies(y_test)[["low","moderate","high"]].values
    #    b) get predicted probabilities
    y_proba = pipe.predict_proba(X_test)
    #    c) compute macro-average AUROC (one-vs-rest)
    macro_auroc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    print(f"Macro AUROC (baseline RF): {macro_auroc:.3f}\n")

    # 7) Confusion matrix + ROC curve plots
    labels = ["low","moderate","high"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion(cm, labels, outpath="plots/confusion_baseline.png")
    plot_roc(pipe, X_test, y_test, labels, outpath="plots/roc_baseline.png")


if __name__ == "__main__":
    main()