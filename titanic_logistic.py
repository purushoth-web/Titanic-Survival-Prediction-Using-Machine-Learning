
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

# --- LOAD YOUR DATA ----

df = pd.read_csv('data/titanic.csv')

#  Use sample data to test right now
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Dataset loaded! Shape:", X.shape)

# SPLIT DATA ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# BUILD PIPELINE ----
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

# TRAIN ----
pipeline.fit(X_train, y_train)
print("Model trained successfully!")

# STEP 5: EVALUATE ----
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("===== ROC-AUC SCORE =====")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

#STEP 6: CROSS VALIDATION ----
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
print(f"\n===== CROSS VALIDATION =====")
print(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# STEP 7: FEATURE IMPORTANCE ----
model = pipeline.named_steps['model']
scaler = pipeline.named_steps['scaler']
feature_names = X.columns
coefficients = pd.Series(model.coef_[0], index=feature_names)
print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(coefficients.abs().sort_values(ascending=False).head(10))


