import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# 1. Read the scaled test inputs and the test target
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]

# 2. Load the trained model and make predictions
model = joblib.load("models/trained_model.pkl")
y_pred = model.predict(X_test)

# 3. Compute the scores
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 4. Save the scores in the metrics folder
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f, indent=4)

# 5. Save the predictions next to the true values
predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
predictions.to_csv("data/predictions.csv", index=False)

print("MSE:", round(mse, 4), " R2:", round(r2, 4))
