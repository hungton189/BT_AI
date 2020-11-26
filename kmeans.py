import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def random_data(n):
    x = np.random.randn(n)*10
    y = np.random.randn(n)*10
    return (x,y)

def random_center(k):
    x_center = np.random.randn(k)*10
    y_center = np.random.randn(k)*10
    return (x_center,y_center)

def update_labels(x, y, x_center,y_center):
    n = len(x)
    k = len(x_center)
    labels = []
    for i in range(n):
        kc_min = np.sqrt((x[i] - x_center[0]) ** 2 + (y[i] - y_center[0]) ** 2)
        vt_temp = 0
        for j in range(k):
            if np.sqrt((x[i] - x_center[j]) ** 2 + (y[i] - y_center[j]) ** 2) < kc_min:
                kc_min = np.sqrt((x[i] - x_center[j]) ** 2 + (y[i] - y_center[j]) ** 2)
                vt_temp = j
        labels.append(vt_temp)
    return labels
def len_cluster(labels,len_center):
    len = np.zeros(len_center)
    for i in range(n):
        if labels[i] == 0:
            len[0] +=1
        elif labels[i] == 1:
            len[1] +=1
        else:
            len[2] +=1
    return len

def kmeans(x,y,x_center,y_center):
    labels = update_labels(x,y,x_center,y_center)
    while True:
        print("while")
        n= len(labels)
        #tính trung bình cộng các điểm dữ liệu
        len_node = len_cluster(labels,len(x_center))
        total = np.zeros(2*len(x_center))
        for i in range(len(labels)):
            if labels[i] == 0:
                total[0] += x[i]
                total[1] += y[i]
            elif labels[i] == 1:
                total[2] += x[i]
                total[3] += y[i]
            else:
                total[4] += x[i]
                total[5] += y[i]
        for i in range(len(x_center)):
            x_center[i] = total[2*i]/len_node[i]
            y_center[i] = total[2*i+1]/len_node[i]
        new_labels = update_labels(x,y,x_center,y_center)
        if np.array_equal(new_labels,labels):
            break
        else:
            labels = new_labels
    return (x_center,y_center,labels)

n=50            #số điểm dữ liệu
k=3             #số cluster

def show(x,y,x_center,y_center,labels):
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for i in range(n):
        if labels[i] == 0:
            x_0.append(x[i])
            y_0.append(y[i])
        elif labels[i] == 1:
            x_1.append(x[i])
            y_1.append(y[i])
        else:
            x_2.append(x[i])
            y_2.append(y[i])
    print(len(labels))
    plt.scatter(x_0,y_0,c="g")
    plt.scatter(x_1,y_1,c="#000")
    plt.scatter(x_2,y_2,c="b")
    plt.scatter(x_center,y_center,marker="o",c="r")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
if __name__ == '__main__':
    x, y = random_data(n)
    x_center, y_center = random_center(k)
    labels = update_labels(x,y,x_center,y_center)
    show(x,y,x_center,y_center,labels)
    x_center, y_center, labels = kmeans(x, y, x_center, y_center)  # labels = update_labels(x,y,x_center,y_center)
    show(x, y, x_center, y_center, labels)
    # print(labels)