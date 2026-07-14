# 🛍️ Shopping Behaviour Analytics Dashboard — with Machine Learning

An end-to-end Data Analyst + Machine Learning project: cleans and analyzes customer
shopping transaction data, surfaces business insights through a 3-page **Power BI**
dashboard, and predicts high-value customers with a **Decision Tree Classifier**.

---

## 📌 Project Overview

Retailers sit on transaction-level data that rarely gets turned into decisions. This
project takes the *Shopping Behaviour Updated* dataset and builds a full analytics
pipeline around it — from raw CSV to a polished, filterable BI dashboard with an
embedded machine learning prediction module — the kind of deliverable a retail
analytics or BI team would actually ship.

## ❓ Problem Statement

The business wants to answer three questions from its shopping transaction data:

1. **What does our customer base look like**, and where is revenue concentrated (which
   categories, products, age groups, genders)?
2. **How do different customer segments behave** — spend, frequency, category
   preference — so marketing can target them differently?
3. **Which customers are "high value"** (high spend + repeat buyers), so the business
   can proactively route them into loyalty or retention campaigns instead of finding
   out after the fact?

## 📊 Dataset Description

**Source:** Shopping Behaviour Updated dataset (`shopping behaviour updated.csv`) —
customer-level retail transactions, 18 columns / ~3,900 records, including:

| Column | Description |
|---|---|
| Customer ID | Unique identifier |
| Age, Gender | Demographics |
| Item Purchased, Category | Product details |
| Purchase Amount (USD) | Transaction value |
| Location, Size, Color, Season | Purchase attributes |
| Review Rating | Customer satisfaction (1–5) |
| Subscription Status | Loyalty program membership |
| Shipping Type, Payment Method | Fulfilment details |
| Discount Applied, Promo Code Used | Promotion flags |
| Previous Purchases | Customer purchase history |
| Frequency of Purchases | Repeat purchase cadence |

> **Note:** the raw CSV wasn't available when this repo was assembled, so
> `Dataset/shopping_behavior_updated.csv` is a schema-matched synthetic sample
> (`Notebook/generate_dataset.py`) generated to mirror the real dataset's columns,
> categories, and distributions. Drop your real `shopping behaviour updated.csv` into
> `Dataset/` with that filename and every script/notebook below runs unchanged against
> it — nothing needs to be modified.

## 🛠️ Technologies Used

- **Python** — Pandas, NumPy (data cleaning & feature engineering)
- **Matplotlib, Seaborn** — exploratory data analysis & static visualizations
- **Scikit-learn** — Decision Tree Classifier, train/test split, evaluation metrics
- **Power BI** — interactive 3-page executive/behavioural/ML dashboard
- **Streamlit** — companion live single-customer prediction app
- **Jupyter Notebook** — analysis narrative and reproducible pipeline

## 🤖 Machine Learning Approach

- **Algorithm:** Decision Tree Classifier (`max_depth=5`, `min_samples_leaf=20`)
- **Target:** `High Value Customer` — engineered label: 1 if a customer's purchase
  amount **and** previous purchase count are both at/above the dataset median, else 0.
  This turns raw transaction fields into an actionable marketing segment.
- **Features:** Age, Gender, Category, Item Purchased, Purchase Amount (USD) —
  label-encoded categoricals + numeric fields
- **Split:** 80/20 train/test, stratified on the target
- **Evaluation:** Accuracy, Confusion Matrix, Precision/Recall/F1 (classification
  report), Feature Importance, and a rendered Decision Tree diagram (see
  `Screenshots/model_*.png` and `Model/model_metrics.json`)

## 📈 Dashboard Features (Power BI — see `Dashboard/POWER_BI_BUILD_GUIDE.md` for the full build spec)

**Page 1 — Executive Overview**
KPI cards (Total Customers, Total Purchase Amount, Average Purchase Amount, Number of
Categories, Most Purchased Item) + gender distribution, purchase trend, category-wise
sales, and top 10 products.

**Page 2 — Customer Behaviour Analysis**
Age group analysis, gender-based purchasing behaviour, category preferences, customer
segmentation (spend x age group), and purchase frequency — all cross-filterable via
slicers (Gender, Age Group, Category, Season, Subscription Status).

**Page 3 — Machine Learning Prediction Dashboard**
Model accuracy card, confusion matrix, classification report, feature importance
chart, and decision tree visualization, plus a scored customer table. A companion
Streamlit app (`Dashboard/streamlit_predict_app.py`) provides true live single-record
prediction, since Power BI can't run inference interactively per click.

## 📈 Results and Insights

| Metric | Value |
|---|---|
| Total Customers | 3,900 |
| Total Purchase Amount | $179,659 |
| Average Purchase Amount | $46.05 |
| Number of Categories | 4 |
| Most Purchased Item | Hoodie |
| Average Review Rating | 3.75 / 5 |
| Subscription Rate | 26.5% |
| **Decision Tree Accuracy** | **76.1%** |

*(Recompute this table after running the pipeline on your real dataset — the numbers
above are from the synthetic sample.)*

Key business takeaways:
- Purchase volume and spend concentrate in the **26–45** age range — the core segment
  for retention campaigns.
- **Clothing** and **Accessories** drive the largest share of category revenue.
- **Purchase Amount** and **Previous Purchases** are the strongest predictors of the
  high-value segment (see `Screenshots/model_feature_importance.png`), meaning the
  model — and the business — should watch repeat-purchase behaviour, not just
  one-off basket size.

## 🖼️ Screenshots

| | |
|---|---|
| ![Gender Distribution](Screenshots/eda_gender_distribution.png) | ![Category Sales](Screenshots/eda_category_sales.png) |
| ![Top 10 Products](Screenshots/eda_top10_products.png) | ![Age Group](Screenshots/eda_age_group.png) |
| ![Heatmap](Screenshots/eda_heatmap_category_age.png) | ![Purchase Frequency](Screenshots/eda_purchase_frequency.png) |
| ![Confusion Matrix](Screenshots/model_confusion_matrix.png) | ![Feature Importance](Screenshots/model_feature_importance.png) |
| ![Decision Tree](Screenshots/model_decision_tree.png) | |

*(Add real Power BI page screenshots here once you've built the `.pbix` following the
build guide — `powerbi_page1_overview.png`, `powerbi_page2_behaviour.png`,
`powerbi_page3_ml.png`.)*

## 📁 Project Structure

```
shopping-behaviour-analytics/
├── Dataset/
│   ├── shopping_behavior_updated.csv      # raw data
│   ├── shopping_behavior_cleaned.csv      # cleaned + feature-engineered (Power BI source)
│   └── shopping_behavior_encoded.csv      # label-encoded (ML training data)
├── Notebook/
│   ├── shopping_behavior_analysis.ipynb   # full cleaning + EDA + ML notebook
│   ├── 01_pipeline.py                     # same pipeline as a standalone script
│   ├── generate_dataset.py                # synthetic dataset generator
│   └── build_notebook.py                  # assembles the .ipynb
├── Model/
│   ├── decision_tree_model.pkl
│   ├── label_encoders.pkl
│   └── model_metrics.json
├── Dashboard/
│   ├── POWER_BI_BUILD_GUIDE.md            # DAX measures + page-by-page build spec
│   └── streamlit_predict_app.py           # companion live-prediction app
├── Screenshots/
│   ├── eda_*.png
│   └── model_*.png
├── README.md
└── requirements.txt
```

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (cleaning, EDA, model training, exports)
cd Notebook
python 01_pipeline.py
# or open shopping_behavior_analysis.ipynb in Jupyter for the narrated version

# 3. Build the Power BI dashboard
# Open Power BI Desktop -> follow Dashboard/POWER_BI_BUILD_GUIDE.md

# 4. (Optional) Run the live prediction companion app
cd Dashboard
streamlit run streamlit_predict_app.py
```

## 👤 Author

[Your Name] — Data Analyst / BI Developer
