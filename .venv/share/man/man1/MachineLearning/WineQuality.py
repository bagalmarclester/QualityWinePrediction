import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv('/home/marc-lester/Downloads/winequality-red-selected-missing.csv')
df.head()

print("Missing values per column:")
print(df.isnull().sum())

df = df.dropna(how="all")
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
df = df.drop_duplicates()

for col in ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"]:
    df = df[df[col] >= 0]

df["target_good"] = (df["quality"] >= 7).astype(int)
df.to_csv("winequality_red_cleaned.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print(df.head())

print("Cleaned dataset shape:", df.shape)
print(df.head())

x = df.drop(columns=["quality", "target_good"])
y = df["target_good"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
}

results = {}
for name, clf in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
y_proba = pipeline.predict_proba(x_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

results[name] = {
    "CV ROC AUC mean": np.mean(cv_auc),
    "Test Accuracy": acc,
    "Test ROC AUC": roc_auc,
    "Confusion Matrix": cm,
    "Classification Report": report,
    "Pipeline": pipeline
}

for name, res in results.items():
    print(f"\n===== {name} =====")
    print("Cross-val ROC AUC mean:", round(res["CV ROC AUC mean"], 4))
    print("Test Accuracy:", round(res["Test Accuracy"], 4))
    print("Test ROC AUC:", round(res["Test ROC AUC"], 4))
    print("Confusion Matrix:\n", res["Confusion Matrix"])
    print("Classification Report:\n", res["Classification Report"])

best_model_name = max(results, key=lambda k: results[k]["Test ROC AUC"])
best_pipeline = results[best_model_name]["Pipeline"]

from sklearn.metrics import classification_report, confusion_matrix

y_proba = best_pipeline.predict_proba(x_test)[:, 1]

threshold = 0.4
y_pred_custom = (y_proba >= threshold).astype(int)

print(f"\n===== {best_model_name} with threshold {threshold} =====")
print(classification_report(y_test, y_pred_custom, digits=4))
print(confusion_matrix(y_test, y_pred_custom))

print(f"\nBest model: {best_model_name}")
joblib.dump(best_pipeline, "wine_quality_best_model.pkl")
print("Saved best model as wine_quality_best_model.pkl")

