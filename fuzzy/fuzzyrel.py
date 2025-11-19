import numpy as np

# ------------------ PART 1: FUZZY SET OPERATIONS ------------------
print("\n===== PART 1: FUZZY SET OPERATIONS =====\n")

A = np.array([0.2, 0.3, 0.4, 0.5])
B = np.array([0.1, 0.2, 0.2, 1.0])
keys = [1, 2, 3, 4]

operations = {
    "Algebraic Sum": np.round(A + B - (A * B), 3),
    "Algebraic Product": np.round(A * B, 3),
    "Bounded Sum": np.round(np.minimum(1, A + B), 3),
    "Bounded Difference": np.round(np.maximum(0, A - B), 3)
}

for op_name, result in operations.items():
    result_dict = {keys[i]: float(result[i]) for i in range(len(keys))}
    # print in the same k/v style you had: value/key
    pairs = ", ".join(f"{v}/{k}" for k, v in result_dict.items())
    print(f"{op_name} = {{ {pairs} }}")

# ------------------ PART 2: FUZZY CARTESIAN PRODUCT ------------------
print("\n===== PART 2: FUZZY CARTESIAN PRODUCT =====\n")

A_membership = np.array([0.3, 0.7, 1.0])
B_membership = np.array([0.4, 0.9])

A_labels = ["a1", "a2", "a3"]
B_labels = ["b1", "b2"]

cartesian_product = np.minimum(A_membership[:, np.newaxis], B_membership)

print("Fuzzy Set A = { " + ", ".join(f"{A_labels[i]}:{A_membership[i]}" for i in range(len(A_labels))) + " }")
print("Fuzzy Set B = { " + ", ".join(f"{B_labels[i]}:{B_membership[i]}" for i in range(len(B_labels))) + " }")

print("\nCartesian Product A × B (rows = A, columns = B):")
print("       " + "   ".join(B_labels))
for i, row in enumerate(cartesian_product):
    print(f"{A_labels[i]}   " + "   ".join(f"{val:.1f}" for val in row))

print("\nStep-by-step μR calculations:")
for i in range(len(A_membership)):
    for j in range(len(B_membership)):
        val = cartesian_product[i, j]
        print(f"μR({A_labels[i]}, {B_labels[j]}) = min( μA({A_labels[i]}) = {A_membership[i]}, μB({B_labels[j]}) = {B_membership[j]} ) = {val}")

# ------------------ PART 3: FUZZY RELATION COMPOSITION ------------------
print("\n===== PART 3: FUZZY RELATION COMPOSITION =====\n")

R = np.array([[0.6, 0.3],
              [0.2, 0.9]])

S = np.array([[1.0, 0.5, 0.3],
              [0.8, 0.4, 0.7]])

X = ["x1", "x2"]
Y = ["y1", "y2"]
Z = ["z1", "z2", "z3"]

print("Fuzzy Relation R:")
print("       " + "   ".join(Y))
for i, row in enumerate(R):
    print(f"{X[i]}   " + "   ".join(f"{val:.1f}" for val in row))

print("\nFuzzy Relation S:")
print("       " + "   ".join(Z))
for i, row in enumerate(S):
    print(f"{Y[i]}   " + "   ".join(f"{val:.1f}" for val in row))

m, n = R.shape
n_check, p = S.shape
if n != n_check:
    raise ValueError("Inner dimensions of R and S do not match for composition.")

# (a) MAX-MIN COMPOSITION
T_max_min = np.zeros((m, p))
print("\n(a) MAX–MIN COMPOSITION T = R ∘ S\n")
for i in range(m):
    for k in range(p):
        min_values = np.minimum(R[i, :], S[:, k])      # vector of mins over y_j
        T_max_min[i, k] = np.max(min_values)
        print(f"T({X[i]}, {Z[k]}) calculation:")
        for j in range(n):
            print(f"    min(R({X[i]},{Y[j]})={R[i,j]}, S({Y[j]},{Z[k]})={S[j,k]}) = {min_values[j]:.3f}")
        print(f"    -> max = {T_max_min[i,k]:.3f}\n")

print("MAX–MIN Composition Matrix:")
print("       " + "   ".join(Z))
for i in range(m):
    print(f"{X[i]}   " + "   ".join(f"{T_max_min[i,j]:.3f}" for j in range(p)))

# (b) MAX-PRODUCT COMPOSITION
T_max_product = np.zeros((m, p))
print("\n(b) MAX–PRODUCT COMPOSITION T = R ∘ S\n")
for i in range(m):
    for k in range(p):
        product_values = R[i, :] * S[:, k]            # vector of products over y_j
        T_max_product[i, k] = np.max(product_values)
        print(f"T({X[i]}, {Z[k]}) calculation:")
        for j in range(n):
            print(f"    R({X[i]},{Y[j]})={R[i,j]} * S({Y[j]},{Z[k]})={S[j,k]} = {product_values[j]:.3f}")
        print(f"    -> max = {T_max_product[i,k]:.3f}\n")

print("MAX–PRODUCT Composition Matrix:")
print("       " + "   ".join(Z))
for i in range(m):
    print(f"{X[i]}   " + "   ".join(f"{T_max_product[i,j]:.3f}" for j in range(p)))

print("\n===== ALL FUZZY OPERATIONS COMPLETED =====\n")