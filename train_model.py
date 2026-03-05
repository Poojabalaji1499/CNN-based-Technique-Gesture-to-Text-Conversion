import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
data = pd.read_csv("gesture_data.csv", header=None)

# Separate features and labels
X = data.iloc[:, :-1]   # first 63 columns
y = data.iloc[:, -1]    # gesture label

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create neural network model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500
)

# Train the model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")