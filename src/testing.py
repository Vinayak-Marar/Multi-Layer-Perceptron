import numpy as np
from mlp import MLP # Assuming mlp.py is in the same directory
import matplotlib.pyplot as plt

def run_square_test():
    """
    Tests the MLP's ability to learn the square function (y = X^2).
    It includes data scaling, training, and inverse scaling for prediction.
    """
    print("--- Starting MLP Square Function Test ---")

    # 1. Generate Training Data (X and Y)
    # X_train: Numbers from 1 to 99, reshaped to a column vector
    X_train_original = np.linspace(1, 100,1000).reshape(-1, 1).astype(np.float64)
    # y_train: The square of X_train
    y_train_original = X_train_original ** 2

    print(f"Original X_train shape: {X_train_original.shape}")
    print(f"Original y_train shape: {y_train_original.shape}")
    print(f"Original X_train sample (first 5): {X_train_original[:5].flatten()}")
    print(f"Original y_train sample (first 5): {y_train_original[:5].flatten()}")

    # 2. Data Scaling (Min-Max Scaling to [0, 1])
    # Store min/max values for both X and Y for consistent scaling/inverse-scaling
    X_min, X_max = X_train_original.min(), X_train_original.max()
    y_min, y_max = y_train_original.min(), y_train_original.max()

    X_train_scaled = (X_train_original - X_min) / (X_max - X_min)
    y_train_scaled = (y_train_original - y_min) / (y_max - y_min)

    print(f"\nScaling parameters: X_min={X_min}, X_max={X_max}, y_min={y_min}, y_max={y_max}")
    print(f"Scaled X_train sample (first 5): {X_train_scaled[:5].flatten()}")
    print(f"Scaled y_train sample (first 5): {y_train_scaled[:5].flatten()}")

    # 3. Initialize MLP
    # Input dimension is 1 (for X)
    # Hidden layers use "leaky-relu" for better gradient flow than sigmoid in regression
    # Output layer uses "linear" for regression task
    mlp = MLP(input_dim=1, structure=[(10,"leaky-relu"),(10, "leaky-relu"),(5,"leaky-relu"), (1, "linear")])
    print("\nMLP initialized with structure: 1 input -> 10 leaky-relu -> 10 leaky-relu -> 1 linear output")

    # 4. Fit and Train MLP
    mlp.fit(X_train_scaled, y_train_scaled)
    print("MLP fitted with scaled training data.")

    epochs = 500 # Number of training iterations
    learning_rate = 0.1 # Learning rate
    mlp.train(epochs, learning_rate)
    print(f"\nMLP training complete after {epochs} epochs.")

    # 5. Make Predictions and Inverse Scale
    test_values_original = np.arange(1,99).reshape(-1, 1).astype(np.float64)
    test_value_predicted =[]
    test_value_true = []
    print("\n--- Making Predictions ---")
    for val_original in test_values_original:
        # Scale the input value for prediction
        val_scaled = (val_original - X_min) / (X_max - X_min)
        
        # Make prediction using the scaled input (mlp.predict expects a 1D array for a single sample)
        predicted_scaled = mlp.predict(val_scaled[0]) 
        
        # Inverse scale the predicted value back to the original range
        predicted_original = predicted_scaled * (y_max - y_min) + y_min
        test_value_predicted.append(predicted_original)
        true_value = val_original[0] ** 2
        test_value_true.append(true_value)
        
        print(f"Input X: {val_original[0]:.2f}")
        print(f"  Scaled Input: {val_scaled[0]:.4f}")
        # Fix: Use .item() to convert 0-dim numpy array to scalar for formatting
        print(f"  Predicted Y (Scaled): {predicted_scaled.item():.4f}") 
        print(f"  Predicted Y (Original): {predicted_original.item():.2f}")
        print(f"  True Y (Original): {true_value:.2f}")
        print(f"  Difference: {abs(predicted_original.item() - true_value):.2f}\n")

    print("--- MLP Square Function Test Complete ---")

    plt.plot(test_values_original,test_value_predicted)
    # plt.scatter(test_values_original,test_value_true)
    plt.show()
if __name__ == "__main__":
    run_square_test()
