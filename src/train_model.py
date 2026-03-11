import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv("../data/housing.csv")

# Separate features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Convert categorical variables
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, predictions, squared=False)
print("RMSE:", rmse)

# Save the trained model
joblib.dump(model, "../models/housing_model.pkl")

print("Model saved to models/housing_model.pkl")
