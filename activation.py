import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Activation Functions
# ----------------------------------------------------
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

activations = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "leaky_relu": (leaky_relu, d_leaky_relu)
}

# ----------------------------------------------------
# ANN with 1 Hidden Layer
# ----------------------------------------------------
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size,
                 activation="relu", lr=0.1):

        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.activation, self.d_activation = activations[activation]
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = len(y)

        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.W2.T * self.d_activation(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=2000):
        losses = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            self.backward(X, y)
        return losses

    def accuracy(self, X, y):
        preds = (self.forward(X) > 0.5).astype(int)
        return np.mean(preds == y)


# ----------------------------------------------------
# Synthetic Dataset (Custom)
# ----------------------------------------------------
def make_dataset(n=200):
    class0 = np.random.randn(n, 2) * 0.5 + np.array([0, 0])
    class1 = np.random.randn(n, 2) * 0.5 + np.array([3, 3])

    X = np.vstack([class0, class1])
    y = np.array([0]*n + [1]*n).reshape(-1,1)

    return X, y

X, y = make_dataset()

plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap="coolwarm")
plt.title("Custom Synthetic Dataset")
plt.show()


# ----------------------------------------------------
# Train ANN using ReLU Activation
# ----------------------------------------------------
nn = SimpleNN(input_size=2, hidden_size=6, output_size=1,
              activation="relu", lr=0.1)

losses = nn.train(X, y, epochs=3000)

print("Training Accuracy:", nn.accuracy(X, y))

plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# ----------------------------------------------------
# Plot Activation Functions
# ----------------------------------------------------
x_vals = np.linspace(-5, 5, 200)
fig, axs = plt.subplots(2, 4, figsize=(16,6))

for i, (name, (f, df)) in enumerate(activations.items()):
    axs[0, i].plot(x_vals, f(x_vals))
    axs[0, i].set_title(name)
    axs[1, i].plot(x_vals, df(x_vals))
    axs[1, i].set_title("d" + name)

plt.tight_layout()
plt.show()