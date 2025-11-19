import numpy as np

def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def bipolar_sigmoid_derivative(y):
    return 0.5 * (1 + y) * (1 - y)


alpha = 0.25
inputs = np.array([-1, 1])
target = 1


W_input_hidden = np.array([
 [0.6, -0.3],
 [-0.1, 0.4],
])


bias_hidden = np.array([0.3, 0.5])
W_hidden_output = np.array([0.4, 0.1])
bias_output = -0.2

print("Initial Weights:")
print("W_input_hidden:\n", W_input_hidden)
print("bias_hidden:", bias_hidden)
print("W_hidden_output:", W_hidden_output)
print("bias_output:", bias_output)
print("="*30)


net_hidden = np.dot(inputs, W_input_hidden) + bias_hidden
out_hidden = bipolar_sigmoid(net_hidden)
net_output = np.dot(out_hidden, W_hidden_output) + bias_output
out_output = bipolar_sigmoid(net_output)


print("\nForward Pass:")
print("Net hidden:", net_hidden)
print("Hidden outputs:", out_hidden)
print("Net output:", net_output)
print("Final output:", out_output)
print("="*30)


error = target - out_output
delta_output = error * bipolar_sigmoid_derivative(out_output)
delta_hidden = bipolar_sigmoid_derivative(out_hidden) * (delta_output * W_hidden_output)


print("\nBackward Pass:")
print("Error:", error)
print("Delta Output:", delta_output)
print("Delta Hidden:", delta_hidden)
print("="*30)


W_hidden_output += alpha * delta_output * out_hidden
bias_output += alpha * delta_output
for i in range(len(inputs)):
    W_input_hidden[i] += alpha * delta_hidden * inputs[i]
bias_hidden += alpha * delta_hidden



print("\nUpdated Weights:")
print("W_input_hidden:\n", W_input_hidden)
print("bias_hidden:", bias_hidden)
print("W_hidden_output:", W_hidden_output)
print("bias_output:", bias_output)
print("="*30)