import numpy as np


def heaviside_step_function(net_input: float) -> int:
    """Heaviside step activation function."""
    return 1 if net_input >= 0 else 0


def get_initial_params(n_features: int):
    """Prompt user for initial weights and bias."""
    try:
        weights = [float(input(f"Enter initial weight w{i+1}: ")) for i in range(n_features)]
        bias = float(input("Enter initial bias b: "))
        return np.array(weights), bias
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None, None


def get_training_data(n_samples: int, n_features: int):
    """Prompt user to input training patterns and target values."""
    X = np.zeros((n_samples, n_features))
    Y = np.zeros(n_samples)

    print("\n--- Enter Patterns ---")
    for i in range(n_samples):
        print(f"\nPattern {i+1}:")
        X[i] = [float(input(f"Enter input x{j+1} for pattern {i+1}: ")) for j in range(n_features)]
        Y[i] = float(input(f"Enter target output y for pattern {i+1}: "))

    return X, Y


def hebbian_learning(W, B, X, Y, n_epochs: int):
    """Perform Hebbian learning over given epochs."""
    for epoch in range(n_epochs):
        print(f"\n===== Epoch {epoch + 1}/{n_epochs} =====")
        for i, (x_pattern, y_target) in enumerate(zip(X, Y)):
            W_old, B_old = W.copy(), B

            # Weighted sum and prediction
            net_input = np.dot(W_old, x_pattern) + B_old
            y_predicted = heaviside_step_function(net_input)

            # Hebbian rule update
            delta_W = x_pattern * y_target
            W = W_old + delta_W
            B = B_old + y_target

            # Epoch details
            print(f"\nPattern {i+1}:")
            print(f"Input (X): {x_pattern}, Target (Y): {y_target}")
            print(f"Old Weights: {W_old}, Old Bias: {B_old}")
            print(f"Net Input: {net_input}, Predicted: {y_predicted}")
            print(f"Weight Change (ΔW): {delta_W}")
            print(f"Updated Weights: {W}, Updated Bias: {B}")

    return W, B


def hebbian_network():
    """Main function to run Hebbian learning demonstration."""
    n_features = 4
    n_samples = 2
    n_epochs = 2

    # Get initial weights and bias
    W, B = get_initial_params(n_features)
    if W is None:
        return

    print("\n--- Initial State ---")
    print(f"Initial Weights: {W}")
    print(f"Initial Bias: {B}\n")

    # Collect training data
    X, Y = get_training_data(n_samples, n_features)

    print("\n--- Data Shapes ---")
    print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")

    # Training
    print("\n--- Hebbian Learning Process ---")
    W, B = hebbian_learning(W, B, X, Y, n_epochs)

    # Final results
    print("\n--- Final State ---")
    print(f"Final Weights: {W}")
    print(f"Final Bias: {B}")


if __name__ == "__main__":
    hebbian_network()
