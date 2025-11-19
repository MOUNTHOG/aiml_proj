
# Train the autoassociative network for input vector 
# [ 1 1 1 1] − and also test the network for the same 
# input vector. Test the autoassociative network with one 
# missing, one mistake, two missing and two mistake 
# entries in test vector.
import numpy as np

# ---------------------------------------
# TRAINING PHASE
# ---------------------------------------

# Input / target pattern
x = np.array([-1, 1, 1, 1])
target = x.copy()

# Weight matrix using autoassociative Hebbian rule
W = np.outer(target, x)

print("Weight Matrix:\n", W)

# ---------------------------------------
# ACTIVATION FUNCTION
# ---------------------------------------

def activation(y):
    return np.where(y > 0, 1, -1)

# ---------------------------------------
# TESTING FUNCTION
# ---------------------------------------

def test_network(test_input, W):
    y_in = np.dot(W, test_input)
    y_out = activation(y_in)
    return y_in, y_out

# ---------------------------------------
# TEST CASES
# ---------------------------------------

test_cases = {
    "Original input": np.array([-1, 1, 1, 1]),

    # One missing values (replace by 0)
    "One missing [0 1 1 1]": np.array([0, 1, 1, 1]),
    "One missing [-1 1 0 1]": np.array([-1, 1, 0, 1]),

    # One mistake (one bit flipped)
    "One mistake [-1 -1 1 1]": np.array([-1, -1, 1, 1]),
    "One mistake [1 1 1 1]": np.array([1, 1, 1, 1]),

    # Two missing values
    "Two missing [0 0 1 1]": np.array([0, 0, 1, 1]),
    "Two missing [-1 0 0 1]": np.array([-1, 0, 0, 1]),

    # Two mistakes
    "Two mistakes [-1 -1 -1 1]": np.array([-1, -1, -1, 1]),
}

# ---------------------------------------
# EXECUTE TESTS
# ---------------------------------------

for desc, vec in test_cases.items():
    print(f"\nTest Case: {desc}")
    y_in, y_out = test_network(vec, W)

    print("Input:", vec)
    print("Net input:", y_in)
    print("Output:", y_out)

    if np.array_equal(y_out, target):
        print("Correct response is obtained.")
    else:
        print("Applying activation gives an incorrect response.")
        print(f"Thus, the network with {desc.lower()} is not recognized.")
