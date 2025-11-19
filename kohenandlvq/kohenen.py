import random
x_len = int(input("Enter the dimention of inputs: "))
y_len = int(input("Enter the number of clusters: "))
n = int(input("Enter the number of inputs: "))

a = 0.5
X = []
for k in range(n):
    inp=[]
    for i in range(x_len):
        inp.append(int(input(f"Enter the value of input {i+1} for input {k+1}: ")))
    X.append(inp)

W = []
for i in range(x_len):
    W.append([])
    for j in range(y_len):
        W[i].append(random.uniform(0,1))
e = 1000

for epoch in range(e):
    for x in X:
        d = []
        for j in range(y_len):
            dist = 0
            for i in range(x_len):
                dist += (x[i]-W[i][j])**2
            d.append(dist**0.5)
        winner = d.index(min(d))
        for i in range(x_len):
            W[i][winner] = W[i][winner] + a * (x[i] - W[i][winner])
    a = a * 0.5

print("Final weights:")
print(W)