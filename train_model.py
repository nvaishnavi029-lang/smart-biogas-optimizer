import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
df = pd.read_csv("dataset_waste.csv")

# ✅ UPDATED FEATURES (cleaned dataset)
features = [
    "Pig Manure (kg)", "Kitchen Food Waste (kg)", "Chicken Litter (kg)",
    "Cassava (kg)", "Bagasse Feed (kg)", "Energy Grass (kg)",
    "Banana Shafts (kg)", "Alcohol Waste (kg)", "Municipal Residue (kg)",
    "Fish Waste (kg)", "Water (L)",
    "Temperature (C)", "Humidity (%)",
    "C/N Ratio", "Digester Temp (C)"
]

X = df[features]
y = df["biogas_production"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Sequential([
    Dense(64, activation='relu', input_dim=len(features)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Evaluate
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("RMSE:", round(rmse, 3))

# Save
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained successfully")