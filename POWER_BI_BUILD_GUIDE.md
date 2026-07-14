# Power BI Dashboard — Build Guide

A ready-to-open `.pbix` can't be generated outside Power BI Desktop itself (it's a
Windows binary application with no scriptable engine available here). This guide gives
you the exact data source, DAX measures, and page-by-page layout so you can assemble the
`.pbix` in **10–15 minutes**. Every visual maps 1:1 to the requirements brief.

## 1. Data Source

1. Open Power BI Desktop → **Get Data → Text/CSV**.
2. Load `Dataset/shopping_behavior_cleaned.csv` (this is the cleaned, feature-engineered
   file from the notebook — it already has `Age Group`, `Spending Segment`, and
   `High Value Customer` columns built in).
3. Also load `Model/model_metrics.json` as a second source (**Get Data → JSON**) — this
   feeds the ML metric cards on Page 3.
4. In Power Query Editor, confirm data types: `Age`, `Purchase Amount (USD)`,
   `Previous Purchases`, `Review Rating` → Decimal/Whole Number; everything else → Text.
   Click **Close & Apply**.

## 2. DAX Measures

Create a new **Measures table** (Modeling → New Table → `Measures = {}`), then add:

```DAX
Total Customers = DISTINCTCOUNT('shopping_behavior_cleaned'[Customer ID])

Total Purchase Amount = SUM('shopping_behavior_cleaned'[Purchase Amount (USD)])

Average Purchase Amount = AVERAGE('shopping_behavior_cleaned'[Purchase Amount (USD)])

Number of Categories = DISTINCTCOUNT('shopping_behavior_cleaned'[Category])

Most Purchased Item =
    CALCULATE(
        SELECTEDVALUE('shopping_behavior_cleaned'[Item Purchased]),
        TOPN(1,
            SUMMARIZE('shopping_behavior_cleaned', 'shopping_behavior_cleaned'[Item Purchased],
                "Cnt", COUNTROWS('shopping_behavior_cleaned')),
            [Cnt], DESC
        )
    )

Avg Review Rating = AVERAGE('shopping_behavior_cleaned'[Review Rating])

Subscription Rate % =
    DIVIDE(
        CALCULATE(COUNTROWS('shopping_behavior_cleaned'),
                   'shopping_behavior_cleaned'[Subscription Status] = "Yes"),
        COUNTROWS('shopping_behavior_cleaned')
    )

High Value Customer Rate % =
    DIVIDE(
        CALCULATE(COUNTROWS('shopping_behavior_cleaned'),
                   'shopping_behavior_cleaned'[High Value Customer] = 1),
        COUNTROWS('shopping_behavior_cleaned')
    )

Repeat Purchase Rate % =
    DIVIDE(
        CALCULATE(COUNTROWS('shopping_behavior_cleaned'),
                   'shopping_behavior_cleaned'[Previous Purchases] > 1),
        COUNTROWS('shopping_behavior_cleaned')
    )
```

## 3. Page 1 — Executive Overview

**Layout:** top row = 5 KPI Cards, middle = 2x2 visual grid.

| Visual | Field(s) | Measure |
|---|---|---|
| Card | — | `Total Customers` |
| Card | — | `Total Purchase Amount` (format: currency) |
| Card | — | `Average Purchase Amount` (format: currency) |
| Card | — | `Number of Categories` |
| Card | — | `Most Purchased Item` |
| Donut Chart | `Gender` | `Total Customers` (Customer distribution by gender) |
| Line Chart | `Season` on X (or add a `Purchase Date` if your real data has one) | `Total Purchase Amount` (Purchase amount trend) |
| Clustered Bar Chart | `Category` on Y | `Total Purchase Amount` on X (Category-wise sales) |
| Bar Chart | `Item Purchased` (Top N filter = 10) | `Total Purchase Amount` (Top 10 products) |

Add a **page-level title** text box: *"Shopping Behaviour — Executive Overview"* with the
project subtitle underneath, and a **slicer** for `Season` / `Location` in the top-right corner.

## 4. Page 2 — Customer Behaviour Analysis

| Visual | Field(s) |
|---|---|
| Clustered Column | `Age Group` (X) vs `Total Customers` (Y) — Age group analysis |
| 100% Stacked Bar | `Gender` vs `Category`, values = count — gender-based category preference |
| Pie Chart | `Category` — category preferences share |
| Matrix / Table | `Spending Segment` x `Age Group`, values = count — customer segmentation |
| Heatmap (Matrix w/ conditional formatting, or use the exported PNG from `Screenshots/eda_heatmap_category_age.png` as an image visual) | `Category` rows, `Age Group` columns |
| Bar Chart | `Frequency of Purchases` — purchase frequency analysis |

**Interactivity:** add slicers for `Gender`, `Age Group`, `Category`, `Season`, and
`Subscription Status` in a slicer panel on the left. Set all visuals to cross-filter
each other (default Power BI behaviour — just confirm **Edit Interactions** isn't
disabled on any visual).

## 5. Page 3 — Machine Learning Prediction Dashboard

Since Power BI can't run a live Python model interactively per-click the way a
Streamlit app can, this page presents the **trained Decision Tree's results** as a
model-performance dashboard (the standard, portfolio-accepted approach for BI tools):

| Visual | Source |
|---|---|
| Card | `Model/model_metrics.json` → `accuracy` field (format as %) |
| Image visual | `Screenshots/model_confusion_matrix.png` |
| Table | Classification report — precision/recall/F1 per class, from `model_metrics.json → classification_report` (paste as a small manual table, 2 rows x 4 columns) |
| Image visual | `Screenshots/model_feature_importance.png` |
| Image visual | `Screenshots/model_decision_tree.png` |
| Table | `High Value Customer` flag alongside `Age`, `Gender`, `Category`, `Item Purchased`, `Purchase Amount (USD)` — lets a viewer browse which real customers the model scored as high-value |

**Optional — live "what-if" prediction:** add 5 **What-If parameters** (Modeling → New
Parameter) for Age, Gender, Category, Item, Purchase Amount, then a DAX measure that
looks up the nearest matching row's `High Value Customer` flag as an approximation, OR
simply link out to the companion **Streamlit prediction app** (see below) for true
live single-record scoring — call it out with a text box: *"For live single-customer
prediction, see the companion Streamlit app in `Dashboard/streamlit_predict_app.py`."*

## 6. Formatting checklist

- Consistent font: **Segoe UI** (Power BI default) throughout, 10-12pt body / 18-24pt titles.
- One accent color for KPI cards, one secondary for charts — pick from the palette used
  in the Screenshots (`#4C72B0` blue / `#DD8452` orange) to keep BI and notebook charts visually consistent.
- Add a **navigation bar**: 3 bookmark-buttons ("Overview", "Customer Behaviour", "ML
  Prediction") on a header pinned to all pages (Insert → Buttons → Blank, then
  Bookmarks pane → Add → assign each button's Action to a page link).
- Title each page with the project name + page purpose; add a small "Last refreshed"
  text box driven by a `NOW()` measure for polish.

## 7. Publish

File → Publish → Power BI Service (needs a free Power BI account) to get a shareable
web link and screenshots for your GitHub README / resume portfolio.
