import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.02, epochs=1000, acceptance_value=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.acceptance_value = acceptance_value

        total_weights = (self.input_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.output_size
        weight_range = 1 / np.sqrt(total_weights)

        self.W1 = np.random.uniform(-weight_range, weight_range, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.uniform(-weight_range, weight_range, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def backward(self, X, y, output):
        m = X.shape[0]

        dZ2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  
        return self.A2


    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            loss = 0.5 * np.mean((output - y.reshape(-1, 1)) ** 2)

            if loss <= self.acceptance_value:
                print(f"Training stopped at epoch {epoch}, Loss: {loss:.4f}")
                break

            self.backward(X, y, output)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1].values  
    targets = data.iloc[:, -1].values  
    return features, targets

if __name__ == "__main__":
    filepath = "concrete_data.csv"
    features, targets = load_data(filepath)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    train_ratio = 0.75
    split_index = int(len(features) * train_ratio)
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]
    input_size = X_train.shape[1]
    hidden_size = 12
    output_size = 1
    learning_rate = 0.02
    epochs = 1000
    acceptance_value = 0.01  

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, epochs, acceptance_value)
    nn.train(X_train, y_train)
    predictions = nn.predict(X_test)
    MSE = 0.5 * np.mean((predictions - y_test) ** 2)
    print(f"MSE: {MSE:.4f}")

    while True:
        user_input = input(
            "Enter new data (cement, water, superplasticizer, age) separated by commas (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        try:
            raw_data = [x.strip() for x in user_input.split(',')]
            print(f"Parsed input: {raw_data}")  
            new_data = np.array([float(x) for x in raw_data]).reshape(1, -1)
            print(f"New data array: {new_data}")  

            if new_data.shape[1] != 4:
                print("Invalid input! Please enter exactly 4 values (cement, water, superplasticizer, age).")
                continue

            normalized_new_data = feature_scaler.transform(new_data)
            print(f"Normalized data: {normalized_new_data}")  

            prediction = nn.predict(normalized_new_data)
            original_prediction = target_scaler.inverse_transform(prediction)

            print(f"Predicted Cement Strength: {original_prediction[0][0]:.6f}")
        except ValueError as e:
            print(f"Invalid input! Error: {e}")


