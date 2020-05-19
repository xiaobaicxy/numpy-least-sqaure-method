# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:47:11 2020
numpy实现最小二乘法
@author: 86186
"""
import numpy as np
import matplotlib.pyplot as plt
def loss(w, b, x, y):
    total_loss = 0.0
    differ = w*x + b - y
    for i in range(len(x)):
        total_loss += differ[i] ** 2
    return total_loss/len(x)

#根据公式计算参数w和b
def compute_parameters(x, y):
    numerator = 0.0 #分子
    denominator = 0.0 #分母
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    w = numerator / denominator
    b = y_mean - w * x_mean
    return w, b
    
x = np.linspace(1,100,200) #在[1,100]区间里生成均匀分布的200个数字
noise = np.random.normal(loc=0.0, scale=20, size=200)#生成均值为0，标准差为20的正态分布数据(200个)
y = 0.8 * x + noise
w, b = compute_parameters(x, y)

plt.figure()
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #plt处理中文，字体为微软雅黑
y_pred = w * x + b
plt.scatter(x, y, c='r') #画散点图
plt.title("x与y的拟合函数")
plt.plot(x, y_pred, c='b') #画直线图
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#矩阵求解法
#计算公式params = (X.T dot X)^-1 dot X.T dot Y
x = np.array(x)
y = np.array(y)
x_features = []
for i in range(len(x)):
    x_features.append(np.array([x[i], 1]).T) #将输入特征加上偏置向并转置成列向量
    
x_features = np.array(x_features) #[200, 2]

xTx = np.dot(x_features.T, x_features) #[2, 2]
xTx_inv = np.linalg.inv(xTx) #[2, 2]
xTx_inv_xT = np.dot(xTx_inv, x_features.T)
params = np.dot(xTx_inv_xT, y)

w,b = params[0], params[1]

plt.figure()
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #plt处理中文，字体为微软雅黑
y_pred = w * x + b
plt.scatter(x, y, c='r') #画散点图
plt.title("x与y的拟合函数")
plt.plot(x, y_pred, c='b') #画直线图
plt.xlabel('x')
plt.ylabel('y')
plt.show()

