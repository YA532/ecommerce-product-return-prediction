
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import load_and_preprocess

# Load processed dataset
df = load_and_preprocess()

X = df.drop(columns=["Return_Status"])
y = df["Return_Status"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Two classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight="balanced"
    )
}

results = {}
best_f1 = -1
best_model = None
best_name = ""

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    results[name] = metrics

    print("\nModel:", name)
    for metric, value in metrics.items():
        print(metric, ":", value)

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = pipeline
        best_name = name

# Save best model and metrics
joblib.dump(best_model, "models/best_model.pkl")

output = {
    "best_model": best_name,
    "model_comparison": results
}

with open("models/metrics.json", "w") as file:
    json.dump(output, file, indent=4)

print("\nBest Model:", best_name)
print("Best F1-score:", best_f1)
print("Model saved to models/best_model.pkl")
print("Metrics saved to models/metrics.json")
