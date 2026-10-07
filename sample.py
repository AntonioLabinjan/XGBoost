import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. Generate synthetic dataset
# ============================================================

np.random.seed(42)

N = 5000

df = pd.DataFrame({
    "amount": np.random.exponential(100, N),
    "hour": np.random.randint(0, 24, N),
    "distance": np.random.exponential(20, N),
    "transactions_last_hour": np.random.poisson(2, N),
    "international": np.random.binomial(1, 0.15, N),
    "device_changed": np.random.binomial(1, 0.10, N),
})


# ============================================================
# 2. Create synthetic fraud rule
# ============================================================

fraud_score = (
    (df["amount"] > 500) * 2
    + (df["distance"] > 100) * 2
    + (df["transactions_last_hour"] > 5) * 2
    + df["international"] * 2
    + df["device_changed"] * 2
    + ((df["hour"] < 5) | (df["hour"] > 23)) * 1
)

# Add some randomness so the problem isn't deterministic
noise = np.random.random(N)

df["fraud"] = (
    (fraud_score + noise > 4)
    .astype(int)
)


print(df.head())
print("\nFraud distribution:")
print(df["fraud"].value_counts())


# ============================================================
# 3. Split features / target
# ============================================================

X = df.drop(columns=["fraud"])
y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# 4. Create XGBoost model
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)


# ============================================================
# 5. Train
# ============================================================

model.fit(
    X_train,
    y_train,
)


# ============================================================
# 6. Predict
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
    )
)


# ============================================================
# 8. Feature importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(
    ascending=False
)

print("\nFeature importance:")
print(importance)


# ============================================================
# 9. Test some new transactions
# ============================================================

new_transactions = pd.DataFrame({
    "amount": [
        20,
        1500,
        300,
        900,
    ],
    "hour": [
        14,
        3,
        18,
        2,
    ],
    "distance": [
        5,
        500,
        10,
        250,
    ],
    "transactions_last_hour": [
        1,
        8,
        2,
        7,
    ],
    "international": [
        0,
        1,
        0,
        1,
    ],
    "device_changed": [
        0,
        1,
        0,
        1,
    ],
})


new_predictions = model.predict(
    new_transactions
)

new_probabilities = model.predict_proba(
    new_transactions
)[:, 1]


print("\nNew transactions:")

for i, (prediction, probability) in enumerate(
    zip(new_predictions, new_probabilities)
):
    label = "FRAUD" if prediction == 1 else "NORMAL"

    print(
        f"Transaction {i + 1}: "
        f"{label} "
        f"(fraud probability = {probability:.3f})"
    )
