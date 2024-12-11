import numpy as np

# Function to input selection parameters
def input_parameters():
    water_quality = float(input("Enter water quality (0 to 1): "))
    proximity_market = float(input("Enter proximity to market (0 to 1): "))
    climate_suitability = float(input("Enter climate suitability (0 to 1): "))
    land_cost = float(input("Enter land cost (0 to 1): "))
    return [water_quality, proximity_market, climate_suitability, land_cost]

# Example dataset: Each row represents a location with different criteria
# Criteria: Water Quality, Proximity to Market, Climate Suitability, Land Cost
data = np.array([[0.8, 0.9, 0.7, 0.6],  # Location 1
                 [0.5, 0.6, 0.8, 0.7],  # Location 2
                 [0.9, 0.8, 0.9, 0.5],  # Location 3
                 [0.6, 0.7, 0.6, 0.8]]) # Location 4

# Labels: 1 indicates suitable, 0 indicates not suitable
labels = np.array([1, 0, 1, 0])

# Normalize the data (optional, depending on your dataset)
data = data / np.max(data, axis=0)

# Add bias term
data = np.hstack((np.ones((data.shape[0], 1)), data))

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            predictions = sigmoid(np.dot(X, self.weights))
            errors = y - predictions
            gradient = np.dot(X.T, errors)
            self.weights += self.learning_rate * gradient

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights)) >= 0.5

# Initialize and train the model
model = LogisticRegression(learning_rate=0.01, epochs=10000)
model.fit(data, labels)

# Predict using the model
predictions = model.predict(data)
print("Predictions:", predictions)

# Evaluate the model
accuracy = np.mean(predictions == labels)
print('Accuracy:', accuracy)

# Input new parameters and predict suitability
new_data = np.array(input_parameters())
new_data = new_data / np.max(new_data)  # Normalize input data
new_data = np.hstack(([1], new_data))   # Add bias term
new_prediction = model.predict(np.array([new_data]))
print("Suitability of new location:", "Suitable" if new_prediction[0] else "Not Suitable")
