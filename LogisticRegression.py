import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data_classification.csv", header=None)
studied = data.values[:, 0]
slept = data.values[:, 1]
result = data.values[:, 2]
true_studied = []
true_slept = []
false_studied = []
false_slept = []
# print(data)
for i in range(len(result)):
    if result[i] == 1:
        true_studied.append(studied[i])
        true_slept.append(slept[i])
    else:
        false_studied.append(studied[i])
        false_slept.append(slept[i])

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def phan_chia(p):
    if p >= 0.5:
        return 1
    else:
        return 0

def predict(features, weights):
    z = np.dot(features,weights)
    return sigmoid(z)

def cost_function(features, labels, weights):
    n = len(labels)
    prediction  = predict(features,weights)

    cost_class1 = -np.log(prediction) * labels              #cost = -log( h(x) ) : label = 1
    cost_class2 = -np.log(1 - prediction) *(1 - labels)     #cost = -log( 1- h(x) ) : label = 0

    cost = cost_class1 + cost_class2
    return cost.sum()/n

def update_weight(features, labels, weights, learning_rate):
    n = len(labels)
    prediction = predict(features,weights)
    gd = np.dot(features.T, prediction-labels)
    gd = gd/n
    gd = gd * learning_rate
    weights = weights - gd

    return weights

def training(features,labels,weights, learning_rate, iter):
    cost_his= []
    for i in range(iter):
        weights = update_weight(features, labels, weights, learning_rate)
        cost = cost_function(features,labels,weights)
        cost_his.append(cost)
    return (weights,cost_his)

weights = [0.1,0.2,0.3]
iter = 800
weights,cost_his = training(data,result,weights,0.5,iter)
print("weight:")
print(weights)
# print("cost_his")
# print(cost_his)

# tính xác suất đúng của hàm training
kt = 0
len = len(result)
for i in range(len):
    if phan_chia(predict(data.values[i,:], weights)) == result[i] :
        kt +=1
max_studied = 0.0
for i in range(len):
    if studied[i] > max_studied:
        max_studied = studied[i]
max = round(max_studied) + 5
print("xác suất đúng của thuật toán:")
print(kt*100/len)


plt.scatter(true_studied, true_slept, marker="o",c="b")
plt.scatter(false_studied, false_slept, marker="s",c="r")
plt.xlabel("Time Studied")
plt.ylabel("Time Slept")
plt.show()