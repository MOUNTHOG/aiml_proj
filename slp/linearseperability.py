import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Perceptron Implementation (for linear separability)
# -------------------------------------------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.lr * (target - y_pred)

                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)


# -------------------------------------------------------
# Dataset A: Linearly Separable
# -------------------------------------------------------
np.random.seed(1)

class0 = np.random.randn(50,2) + np.array([-2, -2])
class1 = np.random.randn(50,2) + np.array([ 2,  2])

X_lin = np.vstack((class0, class1))
y_lin = np.array([0]*50 + [1]*50)

# Train perceptron
p = Perceptron(lr=0.1, epochs=20)
p.fit(X_lin, y_lin)

# Plot linear separable dataset
plt.scatter(X_lin[:,0], X_lin[:,1], c=y_lin, cmap='coolwarm')
plt.title("Linearly Separable Dataset")
plt.show()

# Visualizing decision boundary
x_vals = np.linspace(-5, 5, 100)
y_vals = -(p.weights[0] * x_vals + p.bias) / p.weights[1]

plt.scatter(X_lin[:,0], X_lin[:,1], c=y_lin, cmap='coolwarm')
plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
plt.legend()
plt.title("Perceptron Boundary on Linearly Separable Data")
plt.show()


# -------------------------------------------------------
# Dataset B: Non-Linearly Separable (Circle)
# -------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 100)
circle1 = np.c_[np.cos(theta), np.sin(theta)]     # class 0
circle2 = 2 * np.c_[np.cos(theta), np.sin(theta)] # class 1

X_non = np.vstack((circle1, circle2))
y_non = np.array([0]*100 + [1]*100)

plt.scatter(X_non[:,0], X_non[:,1], c=y_non, cmap='coolwarm')
plt.title("Non-Linearly Separable Dataset")
plt.show()