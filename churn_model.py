import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# ── Load & Preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv("customer_data.csv")

df = pd.get_dummies(df, columns=["contract_type", "payment_method", "internet_service"], drop_first=True)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X_tr, y_train, cv=5, scoring="roc_auc").mean()
    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob, "auc": auc, "cv_auc": cv}
    print(f"\n{'='*40}\n{name}  |  AUC: {auc:.4f}  |  CV-AUC: {cv:.4f}")
    print(classification_report(y_test, y_pred))

# ── Best Model ─────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\n>> Best model: {best_name} (AUC = {best['auc']:.4f})")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Customer Churn Prediction", fontsize=14, fontweight="bold")

# 1. ROC Curves
ax = axes[0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
ax.legend(fontsize=8)

# 2. Confusion Matrix (best model)
ax = axes[1]
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
ax.set(title=f"Confusion Matrix\n{best_name}", xlabel="Predicted", ylabel="Actual")

# 3. Feature Importance (best tree model)
ax = axes[2]
tree_models = {k: v for k, v in results.items() if k != "Logistic Regression"}
best_tree = max(tree_models, key=lambda k: tree_models[k]["auc"])
importances = pd.Series(
    results[best_tree]["model"].feature_importances_, index=X.columns
).nlargest(10).sort_values()
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set(title=f"Top 10 Feature Importances\n{best_tree}", xlabel="Importance")

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n>> Plot saved to churn_analysis.png")
