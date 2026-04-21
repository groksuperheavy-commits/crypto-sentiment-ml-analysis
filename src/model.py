import json
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def run_model(data):
    print("\n===== ML MODEL =====")

   
    candidate_features = [
        "sentiment_value", "Size USD", "Fee", 
        "position", "Side", "Coin"
    ]
    features = [c for c in candidate_features if c in data.columns]

    df = data[features + ["win"]].dropna(subset=["win"]).copy()

    X = df[features]
    y = df["win"]

    numeric_features = [c for c in features if c in ["sentiment_value", "Size USD", "Fee", "Leverage"]]
    categorical_features = [c for c in features if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, preds), 4)
    roc_auc = round(roc_auc_score(y_test, pred_probs), 4)
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds).tolist()

    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.head(20).to_csv("outputs/feature_importance_top20.csv", index=False)

    with open("outputs/model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"ROC-AUC: {roc_auc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC: {roc_auc}")
    print("\nClassification Report:\n", report)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importance": importance_df
    }
