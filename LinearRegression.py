import pandas as pd
import matplotlib.pyplot as plt


def predict(new_radio, weight, bias):
    return  weight * new_radio + bias

def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0.0
    for i in range(n):
        sum_error += (y[i] - (weight * X[i] + bias))**2
    return  sum_error/n

def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * X[i] * (y[i] - (weight*X[i] + bias))
        bias_temp += -2 * (y[i] - (weight*X[i] + bias))
    weight -= (weight_temp/n) * learning_rate
    bias -= (bias_temp/n) * learning_rate
    return (weight, bias)

def training(X, y, weight, bias, learning_rate,iter):
    cos_his = []
    for i in range(iter):
        weight, bias = update_weight(X,y, weight, bias,learning_rate)
        cost = cost_function(X,y,weight,bias)
        cos_his.append(cost)
    return (weight, bias, cos_his)

dataFrame = pd.read_csv("Advertising.csv")
X = dataFrame.values[:, 2]
y = dataFrame.values[:, 4]

iter = 500
weight, bias,cost_his = training(X,y,0.001,0.0001,0.0001,iter)
print("Kết quả")
print(weight)
print(bias)
# print(cost_his)

# vẽ đường thẳng
array_predict = []
n= len(X)
max = 0.0
for i in range(n):
    if X[i] > max:
        max = X[i]
max = round(max) + 5
for i in range(max):
    array_predict.append(weight * i + bias)
array_n = [i for i in range(max)]
plt.plot(array_n, array_predict,c="r")
##############

plt.scatter(X, y, marker="o")
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.show()