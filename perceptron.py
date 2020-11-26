import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("perceptron.csv")
X1 = data.values[:,0]
X2 = data.values[:,1]
y = data.values[:,2]

X1_xanh = []
X2_xanh = []
X1_do = []
X2_do = []
for i in range(len(y)):
    if y[i] == 1:
        X1_xanh.append(X1[i])
        X2_xanh.append(X2[i])
    else:
        X1_do.append(X1[i])
        X2_do.append(X2[i])

def f(weights,X):
    # weight 1xN
    # X      1xN
    # return 1xN
    return np.dot(weights,X) # f(x) = w0 * 1 + w1 * x1 + w2 * x2

def predict(weights,X):
    # return 1 or -1
    return np.sign(f(weights,X))  # hàm dự đoán dấu

def has_converged(weights,X,y):
    #return true or false
    return np.array_equal(predict(weights,X.T)[0], y)    #hàm xác định đã hội tụ hay chưa

def update_weights(weights,xi,yi):
    """
    :param weights: 1xN
    :param xi: 1xN
    :param yi: float
    :return: 1xN
    """
    return weights + xi * yi

def training(X,weights,y):
    n = len(y)
    X = X.T
    weights = weights.T
    m = 0
    while True:
        m +=1
        for i in range(n):
            if predict(weights, X[i])[0] != y[i]:
                weights = update_weights(weights,X[i],y[i])
        if m>2000:
            print("không có kết quả,kiểm tra lại dữ liệu")
            break
        if has_converged(weights,X,y):
            break
    return weights

X = np.row_stack((X1,X2))
X = np.row_stack((np.ones((1,len(X1))),X))
weights = np.random.randn(3, 1)                     #w0 w1 w2
# print(X)
weights = training(X,weights,y)[0]
print(weights)

# vẽ đường phân cách
X2_predict = []
max = 0.0
for i in range(len(X1)):
    if X1[i] > max:
        max = X1[1]
max = round(max) + 5
for i in range(max):
    X2_tt = (-weights[0] - weights[1] * i )/(weights[2])
    X2_predict.append(X2_tt)
array_n = [i for i in range(max)]
plt.plot(array_n,X2_predict,c="g")
#################

##########data
plt.scatter(X1_xanh,X2_xanh,marker="o",c="b")
plt.scatter(X1_do,X2_do,marker="s",c="r")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
