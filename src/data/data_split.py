import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Read the raw data
df = pd.read_csv("data/raw_data/raw.csv")

# 2. Remove the date column (it is text, the model cannot use it)
df = df.drop(columns=["date"])

# 3. Separate the inputs (X) from the target (y)
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# 4. Split: 80 % training, 20 % test (random_state makes it repeatable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Save the four datasets
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split done. Train:", X_train.shape, "Test:", X_test.shape)
