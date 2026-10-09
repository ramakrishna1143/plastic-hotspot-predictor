import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv("data/plastic_waste_data.csv")

# Input features
X = data[
    [
        "people",
        "shops",
        "bins",
        "waste_level",
        "collection_frequency"
    ]
]

# Target
y = data["risk"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)


# Save model
joblib.dump(model, "plastic_hotspot_model.pkl")

print("Model saved as plastic_hotspot_model.pkl")