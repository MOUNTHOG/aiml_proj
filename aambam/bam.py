# Construct and test a BAM network to associate letters 
# E and F with simple bipolar input–output vectors. The 
# target output for E is (–1, 1) and for F is (1, 1). The dis
# play matrix size is 5 3× . The input patters are
# * * *
#  * · ·
#  * * *
#  * · ·
#  * * *
#  * * *
#  * * *

#  * · ·
#  * · ·
#  * · ·
# “E” “F”
#  Targetoutput( 1, 1) (1, 1)

import numpy as np

# -----------------------------------------------------------
# Function to print letter patterns
# -----------------------------------------------------------
def print_letter(matrix, title=""):
    print(title)
    for row in matrix:
        print(" ".join("*" if val == 1 else "." for val in row))
    print()

# -----------------------------------------------------------
# Bipolar activation function
# -----------------------------------------------------------
def activation(vec):
    return np.where(vec > 0, 1, np.where(vec < 0, -1, 0))

# -----------------------------------------------------------
# INPUT PATTERNS (5×3 LETTER MATRICES)
# -----------------------------------------------------------

E_matrix = np.array([
    [1,  1,  1],
    [1, -1, -1],
    [1,  1,  1],
    [1, -1, -1],
    [1,  1,  1]
])

F_matrix = np.array([
    [1,  1,  1],
    [1,  1,  1],
    [1, -1, -1],
    [1, -1, -1],
    [1, -1, -1]
])

# Flatten input patterns to vectors
X_E = E_matrix.flatten()
X_F = F_matrix.flatten()

# Display the input patterns
print_letter(E_matrix, "Letter E (input X_E):")
print("X_E vector:", X_E, "\n")

print_letter(F_matrix, "Letter F (input X_F):")
print("X_F vector:", X_F, "\n")

# -----------------------------------------------------------
# TARGET OUTPUT PATTERNS
# -----------------------------------------------------------
Y_E = np.array([-1, 1])
Y_F = np.array([1, 1])

print("Target output for E (Y_E):", Y_E)
print("Target output for F (Y_F):", Y_F, "\n")

# -----------------------------------------------------------
# BAM WEIGHT MATRIX CONSTRUCTION
# W = Σ (X_p^T × Y_p)
# -----------------------------------------------------------

W1 = np.outer(X_E, Y_E)
W2 = np.outer(X_F, Y_F)
W  = W1 + W2

print("W1 = X_E^T * Y_E:\n", W1, "\n")
print("W2 = X_F^T * Y_F:\n", W2, "\n")
print("Final Weight Matrix W = W1 + W2:\n", W, "\n")

# -----------------------------------------------------------
# TESTING X → Y
# -----------------------------------------------------------

print("Testing X → Y")

for name, x, target in zip(["E", "F"], [X_E, X_F], [Y_E, Y_F]):
    net_in = np.dot(x, W)
    y_out = activation(net_in)

    print(f"\nTest Pattern {name}:")
    print("Input X:", x)
    print("Net input (X·W):", net_in)
    print("Output after activation:", y_out)
    print("Target Y:", target)

    if np.array_equal(y_out, target):
        print("Hence, correct response is obtained.")

# -----------------------------------------------------------
# TESTING Y → X
# -----------------------------------------------------------

print("\nTesting Y → X")

W_T = W.T  # transpose for reverse mapping

for name, y, x in zip(["E", "F"], [Y_E, Y_F], [X_E, X_F]):
    net_in = np.dot(y, W_T)
    x_out = activation(net_in)

    print(f"\nTest Target {name}:")
    print("Input Y:", y)
    print("Net input (Y·W^T):", net_in)
    print("Output X after activation (vector):", x_out)

    print("\nReconstructed 5×3 pattern:")
    print_letter(x_out.reshape(5, 3), f"Recalled {name}")

    if np.array_equal(x_out, x):
        print("Correct response is obtained.")

print("\nThus, a BAM network has been constructed and tested in both directions (X→Y and Y→X).")