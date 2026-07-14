"""
Generates a synthetic version of the 'Shopping Behaviour Updated' dataset
(same schema as the well-known Kaggle Customer Shopping Trends dataset)
with 3900 rows, since the original raw CSV was not available.

If you have the real shopping_behavior_updated.csv, just drop it into the
Dataset/ folder with this exact name and re-run 01_data_cleaning_eda.py —
every downstream script reads from that file path only, nothing else
needs to change.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3900

categories = {
    "Clothing": ["Blouse", "Sweater", "Jeans", "Shirt", "Shorts", "Dress",
                 "Skirt", "Pants", "T-shirt", "Hoodie"],
    "Footwear": ["Sandals", "Sneakers", "Shoes", "Boots", "Slippers"],
    "Outerwear": ["Coat", "Jacket"],
    "Accessories": ["Handbag", "Sunglasses", "Jewelry", "Scarf", "Hat",
                     "Backpack", "Belt", "Gloves"],
}
all_items = [item for items in categories.values() for item in items]
item_to_cat = {item: cat for cat, items in categories.items() for item in items}

locations = ["California", "Texas", "New York", "Florida", "Illinois", "Ohio",
             "Pennsylvania", "Georgia", "North Carolina", "Michigan",
             "Washington", "Arizona", "Massachusetts", "Colorado", "Virginia",
             "New Jersey", "Tennessee", "Indiana", "Missouri", "Wisconsin"]

colors = ["Black", "White", "Grey", "Blue", "Red", "Green", "Yellow", "Purple",
          "Pink", "Orange", "Brown", "Beige", "Maroon", "Charcoal", "Olive"]
sizes = ["S", "M", "L", "XL"]
seasons = ["Spring", "Summer", "Fall", "Winter"]
shipping = ["Express", "Free Shipping", "Next Day Air", "Standard",
            "Store Pickup", "2-Day Shipping"]
payment = ["Credit Card", "Bank Transfer", "Cash", "PayPal", "Debit Card", "Venmo"]
frequency = ["Weekly", "Fortnightly", "Bi-Weekly", "Monthly", "Quarterly",
             "Every 3 Months", "Annually"]

gender = rng.choice(["Male", "Female"], size=N, p=[0.68, 0.32])
age = rng.integers(18, 71, size=N)
item = rng.choice(all_items, size=N)
category = [item_to_cat[i] for i in item]

base_price = {"Clothing": 45, "Footwear": 55, "Outerwear": 75, "Accessories": 35}
purchase_amount = np.array([
    max(20, int(rng.normal(base_price[c], 15))) for c in category
])

review_rating = np.round(rng.uniform(2.5, 5.0, size=N), 1)
subscription = rng.choice(["Yes", "No"], size=N, p=[0.27, 0.73])
discount = rng.choice(["Yes", "No"], size=N, p=[0.43, 0.57])
promo = discount  # promo correlates with discount, matching real dataset pattern
previous_purchases = rng.integers(1, 51, size=N)

df = pd.DataFrame({
    "Customer ID": np.arange(1, N + 1),
    "Age": age,
    "Gender": gender,
    "Item Purchased": item,
    "Category": category,
    "Purchase Amount (USD)": purchase_amount,
    "Location": rng.choice(locations, size=N),
    "Size": rng.choice(sizes, size=N, p=[0.2, 0.4, 0.3, 0.1]),
    "Color": rng.choice(colors, size=N),
    "Season": rng.choice(seasons, size=N),
    "Review Rating": review_rating,
    "Subscription Status": subscription,
    "Shipping Type": rng.choice(shipping, size=N),
    "Discount Applied": discount,
    "Promo Code Used": promo,
    "Previous Purchases": previous_purchases,
    "Payment Method": rng.choice(payment, size=N),
    "Frequency of Purchases": rng.choice(frequency, size=N),
})

# introduce a few realistic messiness artifacts for the cleaning step to handle
dup_idx = rng.choice(N, size=15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)          # duplicates
na_idx = rng.choice(df.index, size=25, replace=False)
df.loc[na_idx, "Review Rating"] = np.nan                          # missing values
na_idx2 = rng.choice(df.index, size=10, replace=False)
df.loc[na_idx2, "Color"] = np.nan

df.to_csv("/home/claude/project/Dataset/shopping_behavior_updated.csv", index=False)
print("Saved:", df.shape)
