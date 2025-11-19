
import numpy as np

# -----------------------------------------
# McCulloch–Pitts Neuron
# -----------------------------------------
def mp_neuron(inputs, weights, threshold):
    return 1 if np.dot(inputs, weights) >= threshold else 0


# -----------------------------------------
# HALF ADDER (4 NEURONS)
# -----------------------------------------
def half_adder(A, B):
    # XOR = A·¬B  +  ¬A·B
    N1 = mp_neuron([A, B], [ 1, -1], 0.5)   # A·¬B
    N2 = mp_neuron([A, B], [-1,  1], 0.5)   # ¬A·B
    SUM = mp_neuron([N1, N2], [1, 1], 1)    # OR(N1, N2)

    # CARRY = A AND B
    CARRY = mp_neuron([A, B], [1, 1], 2)

    return SUM, CARRY


# -----------------------------------------
# FULL ADDER (7 NEURONS)
# -----------------------------------------
def full_adder(A, B, Cin):
    # XOR1 = A XOR B
    N1 = mp_neuron([A, B], [ 1, -1], 0.5)
    N2 = mp_neuron([A, B], [-1,  1], 0.5)
    XOR1 = mp_neuron([N1, N2], [1, 1], 1)

    # SUM = XOR1 XOR Cin
    N3 = mp_neuron([XOR1, Cin], [ 1, -1], 0.5)
    N4 = mp_neuron([XOR1, Cin], [-1,  1], 0.5)
    SUM = mp_neuron([N3, N4], [1, 1], 1)

    # CARRY = majority(A, B, Cin)
    CARRY = mp_neuron([A, B, Cin], [1, 1, 1], 2)

    return SUM, CARRY


# -----------------------------------------
# TESTING
# -----------------------------------------
print("=== HALF ADDER ===")
for A in [0, 1]:
    for B in [0, 1]:
        SUM, CARRY = half_adder(A, B)
        print(f"A={A}, B={B}  ->  SUM={SUM}, CARRY={CARRY}")

print("\n=== FULL ADDER ===")
for A in [0, 1]:
    for B in [0, 1]:
        for Cin in [0, 1]:
            SUM, CARRY = full_adder(A, B, Cin)
            print(f"A={A}, B={B}, Cin={Cin}  ->  SUM={SUM}, CARRY={CARRY}")
