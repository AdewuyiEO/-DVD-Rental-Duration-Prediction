# -DVD-Rental-Duration-Prediction
Machine learning project predicting DVD rental duration using regression models. Achieved MSE = 2.22 with RandomForestRegressor.
# Procedures Taken
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv("rental_info.csv")

# 2. Create target variable (rental_days)
df["rental_date"] = pd.to_datetime(df["rental_date"])
df["return_date"] = pd.to_datetime(df["return_date"])
df["rental_days"] = (df["return_date"] - df["rental_date"]).dt.days

# 3. Convert 'special_features' into dummy variables
special_features_dummies = df["special_features"].str.get_dummies(sep=',')
df = pd.concat([df, special_features_dummies], axis=1)

# 4. Define features and target
X = df.drop(columns=["rental_date", "return_date", "rental_days", "special_features"])
y = df["rental_days"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestRegressor(max_depth=10, n_estimators=51, random_state=9)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
