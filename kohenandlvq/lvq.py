import random

X_len = int(input("Enter the dimension of input: "))
y_len = int(input("Enter the number of clusters: "))
n = int(input("Enter the number of training pairs: "))

X = []
t = []
W=[]
for i in range(n):
    temp = []
    print(f"Enter input {i+1}")
    for j in range(X_len):
        temp.append(int(input(f"x[{j}] = ")))
    X.append(temp)
    t.append(int(input("this belongs to cluster (0 based indexing): ")))

for i in range(X_len):
    row = []
    for j in range(y_len):
        row.append(random.uniform(0,1))
    W.append(row)

a = 0.5

for epoch in range(1000):
    for k in range(n):
        d = []
        for j in range(y_len):
            dist = 0
            for i in range(X_len):
                dist += (X[k][i] - W[i][j]) ** 2
            d.append(dist ** 0.5)
        winner = d.index(min(d))

        if winner == t[k]:
            for i in range(X_len):
                W[i][winner] = W[i][winner] + (a * (X[k][i] - W[i][winner]))
        
        else:
            for i in range(X_len):
                W[i][winner] = W[i][winner] - (a * (X[k][i] - W[i][winner]))
    
    a = 0.5 * a

print(W)
        

        