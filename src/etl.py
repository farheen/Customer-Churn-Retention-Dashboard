import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Basic cleaning
df = df.dropna()

# 3. Encode Churn as binary
df["Churn_Flag"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# 4. Encode categorical variables (for churn probability model later)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("customerID")  # keep ID as is

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 5. Save processed file
df.to_csv("data/processed_churn.csv", index=False)

print("✅ Processed churn dataset saved in /data/processed_churn.csv")

