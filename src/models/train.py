import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. Read the scaled training inputs and the training target
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

# 2. Load the best parameters found by the grid search
best_params = joblib.load("models/best_params.pkl")

# 3. Train one model with these parameters on the whole training set
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# 4. Save the trained model
joblib.dump(model, "models/trained_model.pkl")

print("Model trained with:", best_params)
