"""
Shopping Behaviour Updated - Data Cleaning, EDA, and Decision Tree Classifier
--------------------------------------------------------------------------
Produces:
  Dataset/shopping_behavior_cleaned.csv      -> cleaned, Power-BI-ready data
  Dataset/shopping_behavior_encoded.csv      -> label-encoded data used for ML
  Model/decision_tree_model.pkl              -> trained classifier
  Model/label_encoders.pkl                   -> encoders for the Streamlit-style
                                                 prediction module / re-use
  Model/model_metrics.json                   -> accuracy, classification report,
                                                 confusion matrix
  Screenshots/*.png                          -> EDA + model evaluation charts
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

sns.set_theme(style="whitegrid")
BASE = "/home/claude/project"

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
df = pd.read_csv(f"{BASE}/Dataset/shopping_behavior_updated.csv")
print("Raw shape:", df.shape)

# ----------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

print("\nMissing values before treatment:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Numeric -> median, Categorical -> mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for c in num_cols:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].mode()[0])

assert df.isnull().sum().sum() == 0
print("\nMissing values after treatment: 0")

# consistent formatting
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip()

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING (used across EDA / dashboard / model)
# ----------------------------------------------------------------------
bins = [17, 25, 35, 45, 55, 71]
labels = ["18-25", "26-35", "36-45", "46-55", "56-70"]
df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

spend_bins = [0, 30, 60, 90, 1000]
spend_labels = ["Low", "Medium", "High", "Premium"]
df["Spending Segment"] = pd.cut(df["Purchase Amount (USD)"], bins=spend_bins,
                                 labels=spend_labels)

# Target for the Decision Tree: "High Value Customer" (business-relevant
# classification: is this purchase from a customer likely to spend big /
# be a repeat, discount-driven buyer worth targeting with loyalty offers)
df["High Value Customer"] = (
    (df["Purchase Amount (USD)"] >= df["Purchase Amount (USD)"].median()) &
    (df["Previous Purchases"] >= df["Previous Purchases"].median())
).astype(int)

df.to_csv(f"{BASE}/Dataset/shopping_behavior_cleaned.csv", index=False)
print("\nCleaned dataset saved. Shape:", df.shape)

# ----------------------------------------------------------------------
# 4. EDA CHARTS  (also feed the dashboard's "screenshots")
# ----------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{BASE}/Screenshots/{name}", dpi=150)
    plt.close()

# Gender distribution
plt.figure(figsize=(5, 5))
df["Gender"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90,
                                      colors=["#4C72B0", "#DD8452"])
plt.ylabel("")
plt.title("Customer Distribution by Gender")
savefig("eda_gender_distribution.png")

# Category-wise sales
plt.figure(figsize=(7, 4.5))
cat_sales = df.groupby("Category")["Purchase Amount (USD)"].sum().sort_values(ascending=False)
sns.barplot(x=cat_sales.values, y=cat_sales.index, palette="viridis")
plt.xlabel("Total Purchase Amount (USD)")
plt.title("Category-wise Sales Analysis")
savefig("eda_category_sales.png")

# Top 10 purchased products
plt.figure(figsize=(7, 4.5))
top_items = df["Item Purchased"].value_counts().head(10)
sns.barplot(x=top_items.values, y=top_items.index, palette="magma")
plt.xlabel("Number of Purchases")
plt.title("Top 10 Purchased Products")
savefig("eda_top10_products.png")

# Age group analysis
plt.figure(figsize=(7, 4.5))
sns.countplot(data=df, x="Age Group", order=labels, palette="crest")
plt.title("Customer Count by Age Group")
savefig("eda_age_group.png")

# Gender-based purchasing behaviour (avg spend)
plt.figure(figsize=(6, 4.5))
sns.barplot(data=df, x="Gender", y="Purchase Amount (USD)", palette="pastel",
            estimator=np.mean)
plt.title("Average Purchase Amount by Gender")
savefig("eda_gender_spend.png")

# Heatmap: Category vs Age Group purchase count
plt.figure(figsize=(8, 5))
pivot = pd.crosstab(df["Category"], df["Age Group"])
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Heatmap: Category vs Age Group")
savefig("eda_heatmap_category_age.png")

# Purchase frequency analysis
plt.figure(figsize=(7, 4.5))
freq_order = df["Frequency of Purchases"].value_counts().index
sns.countplot(data=df, y="Frequency of Purchases", order=freq_order, palette="flare")
plt.title("Purchase Frequency Analysis")
savefig("eda_purchase_frequency.png")

print("\nEDA charts saved to Screenshots/")

# ----------------------------------------------------------------------
# 5. ENCODING FOR ML
# ----------------------------------------------------------------------
model_df = df.copy()
encoders = {}
encode_cols = ["Gender", "Category", "Item Purchased"]

for c in encode_cols:
    le = LabelEncoder()
    model_df[c + " Encoded"] = le.fit_transform(model_df[c])
    encoders[c] = le

model_df.to_csv(f"{BASE}/Dataset/shopping_behavior_encoded.csv", index=False)

# ----------------------------------------------------------------------
# 6. DECISION TREE CLASSIFIER
# ----------------------------------------------------------------------
feature_cols = ["Age", "Gender Encoded", "Category Encoded",
                 "Item Purchased Encoded", "Purchase Amount (USD)"]
X = model_df[feature_cols]
y = model_df["High Value Customer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Standard", "High Value"],
                                output_dict=True)

print(f"\nDecision Tree Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Standard", "High Value"]))

# Confusion matrix plot
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Standard", "High Value"],
            yticklabels=["Standard", "High Value"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
savefig("model_confusion_matrix.png")

# Feature importance
plt.figure(figsize=(7, 4.5))
importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()
importance.plot.barh(color="#55A868")
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
savefig("model_feature_importance.png")

# Decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_cols, class_names=["Standard", "High Value"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree Visualization (top 3 levels shown)")
savefig("model_decision_tree.png")

# ----------------------------------------------------------------------
# 7. SAVE MODEL + METRICS
# ----------------------------------------------------------------------
with open(f"{BASE}/Model/decision_tree_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open(f"{BASE}/Model/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

metrics = {
    "accuracy": round(acc, 4),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_importance": importance.to_dict(),
    "features_used": feature_cols,
    "target": "High Value Customer (1 = high value, 0 = standard)",
}
with open(f"{BASE}/Model/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nModel + metrics saved to Model/")
print("\nPipeline complete.")
