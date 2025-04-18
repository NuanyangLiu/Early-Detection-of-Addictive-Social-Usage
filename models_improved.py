import pandas as pd
import numpy as np
import contextlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# require: pip install category_encoders imbalanced-learn
from category_encoders import CatBoostEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

# ——————————————————————————————————————————————————————————
# 1) Preprocessing helpers
# ——————————————————————————————————————————————————————————
def convert_watch_time(df, time_col="Watch Time"):
    dt = pd.to_datetime(df[time_col].astype(str).str.strip(), format="%I:%M %p")
    df["Watch Period"] = dt.dt.hour.map(
        lambda h: "morning"   if 5 <= h < 12 else
                  "afternoon" if 12 <= h < 17 else
                  "evening"   if 17 <= h < 21 else
                  "night"
    )
    return df.drop(columns=[time_col])

def feature_prune(df):
    return df.drop(columns=[
        "UserID", "Video ID", "DeviceType", "OS",
        "Self Control", "ProductivityLoss", "Satisfaction"
    ])

def create_labels(df):
    df["AddictionBand"] = df["AddictionLevel"].map(
        lambda x: "low" if x <= 1 else ("moderate" if x <= 4 else "high")
    )
    return df.drop(columns=["AddictionLevel"])

# ——————————————————————————————————————————————————————————
# 2) Load & preprocess
# ——————————————————————————————————————————————————————————
df = pd.read_csv("data/social_media_dataset.csv")
df = convert_watch_time(df)
df = feature_prune(df)
df = create_labels(df)

# 3) Interaction features
df["avg_session_length"] = df["Total Time Spent"] / df["Number of Sessions"]
df["avg_video_time"]      = df["Time Spent On Video"] / df["Number of Videos Watched"]
df["engagement_rate"]     = df["Engagement"] / df["Number of Videos Watched"]

# 4) Encode labels as integers 0=low,1=moderate,2=high
le = LabelEncoder()
df["y"] = le.fit_transform(df["AddictionBand"])

# 5) Train/test split
X = df.drop(columns=["AddictionBand","y","Frequency"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 6) Columns & SMOTE indices
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
cat_idxs = [X.columns.get_loc(c) for c in cat_cols]

# 7) Preprocessor & SMOTE‑NC
preprocessor = ColumnTransformer([
    ("cat", CatBoostEncoder(), cat_cols),
    ("num", StandardScaler(),    num_cols),
], remainder="drop")

smote = SMOTENC(categorical_features=cat_idxs, random_state=42)

# ——————————————————————————————————————————————————————————
# A) Well‑tuned GBDT pipeline
# ——————————————————————————————————————————————————————————
pipe_gbdt = ImbPipeline([
    ("pre",   preprocessor),
    ("smote", smote),
    ("clf",   HistGradientBoostingClassifier(random_state=42))
])

param_dist_gbdt = {
    "clf__max_iter":          np.linspace(100, 1000, 10, dtype=int),
    "clf__max_depth":         [None] + list(range(5, 31, 5)),
    "clf__learning_rate":     [0.01, 0.05, 0.1],
    "clf__max_leaf_nodes":    [15, 31, 63],
    "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rand_gbdt = RandomizedSearchCV(
    pipe_gbdt,
    param_distributions=param_dist_gbdt,
    n_iter=50,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# ——————————————————————————————————————————————————————————
# Redirect verbose CV logs to file
# ——————————————————————————————————————————————————————————
with open("cv_verbose.log", "w") as log_f:
    with contextlib.redirect_stdout(log_f):
        rand_gbdt.fit(X_train, y_train)

# ——————————————————————————————————————————————————————————
# Report GBDT performance
# ——————————————————————————————————————————————————————————
print("=== GBDT Results ===")
print("Best CV f1_macro:", rand_gbdt.best_score_)
print("Best params:", rand_gbdt.best_params_, "\n")

# Test‑set classification
y_pred_gbdt = rand_gbdt.predict(X_test)
print("GBDT Classification report on test set:\n")
print(classification_report(
    y_test, y_pred_gbdt,
    target_names=le.classes_, digits=3
))

# Macro AUROC
y_proba_gbdt = rand_gbdt.predict_proba(X_test)
roc_macro_gbdt = roc_auc_score(
    y_test, y_proba_gbdt,
    multi_class="ovr", average="macro"
)
print(f"Macro AUROC (GBDT): {roc_macro_gbdt:.3f}\n")

# Confusion matrix – GBDT
cm = confusion_matrix(y_test, y_pred_gbdt, labels=[0,1,2])
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix – GBDT")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], le.classes_)
plt.yticks([0,1,2], le.classes_)
th = cm.max()/2
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j]>th else "black")
plt.colorbar()
plt.tight_layout()
plt.savefig("plots/confusion_gbdt.png")
plt.close()

# ——————————————————————————————————————————————————————————
# B) Cost‑sensitive stacking (RF + LR meta‑model)
# ——————————————————————————————————————————————————————————
base_learners = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("lr", LogisticRegression(class_weight={0:3,1:1,2:1}, max_iter=1000))
]
stack_cs = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(
        class_weight={0:3,1:1,2:1}, max_iter=1000
    ),
    cv=5, n_jobs=-1, passthrough=False
)

pipe_cs = ImbPipeline([
    ("pre",   preprocessor),
    ("smote", smote),
    ("clf",   stack_cs)
])
pipe_cs.fit(X_train, y_train)

# Test‑set classification
y_pred_cs = pipe_cs.predict(X_test)
print("=== Cost‑sensitive Stacking ===")
print(classification_report(
    y_test, y_pred_cs,
    target_names=le.classes_, digits=3
))

# Macro AUROC for stacking
y_proba_cs = pipe_cs.predict_proba(X_test)
roc_macro_cs = roc_auc_score(
    y_test, y_proba_cs,
    multi_class="ovr", average="macro"
)
print(f"Macro AUROC (Stacking): {roc_macro_cs:.3f}\n")

# Confusion matrix – Stacking
cm2 = confusion_matrix(y_test, y_pred_cs, labels=[0,1,2])
plt.figure(figsize=(6,5))
plt.imshow(cm2, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix – Cost‑Sensitive Stacking")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], le.classes_)
plt.yticks([0,1,2], le.classes_)
th2 = cm2.max()/2
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm2[i,j], ha="center", va="center",
                 color="white" if cm2[i,j]>th2 else "black")
plt.colorbar()
plt.tight_layout()
plt.savefig("plots/confusion_cs.png")
plt.close()


