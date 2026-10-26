import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Read the training and test inputs
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# 2. Learn mean and std from the TRAINING data only, then apply to both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Turn the results back into tables with the original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 4. Save the two scaled datasets
X_train_scaled.to_csv("data/processed_data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed_data/X_test_scaled.csv", index=False)

print("Scaling done.")
