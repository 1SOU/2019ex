# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:02:51 2019

@author: Yisoul
"""

#==============================================================================
# for itera in range(n_iterations):
#     gradients= 2/m*(X_bt.dot(X.dot(theta-y))
#     theta = theta - eta*gradients
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import xlrd

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
m=100 
theta=np.random.randn(2,1)
print("初始theta:",theta)
def gradients():
    eta = 0.12 # 学习率
    n_iterations = 1000
    m = 100
    theta=np.random.randn(2,1)
    plt.plot(X,y,'b.')
    for iteration in range(1,n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients 
        y_predict=X_new_b.dot(theta)
        plt.plot(X_new,y_predict,'g-')
    plt.show()
   
def random_gradient():
    plt.plot(X,y,'b.')
    n_epochs = 50
    t0,t1= 5,50
    def learn_schedule(t):
        return t0/(t+t1)
    theta = np.random.randn(2,1)
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index= np.random.randint(m)
            xi= X_b[random_index:random_index+1] # 抽取出的一列，仍是二维数组形式
            #xi= X_b[random_index] 抽取一列，结果为一维数组
            yi= y[random_index:random_index+1]
            gradients= 2*xi.T.dot(xi.dot(theta)-yi)#只对一个样本计算梯度
            eta= learn_schedule(epoch*m+i)
            #控制学习率，随着迭代次数减小
            theta=theta-eta*gradients
            
            y_predict=X_new_b.dot(theta)
            plt.plot(X_new,y_predict,'g-')
            
    print("训练结果：",theta)

def


    
    
    
    
    
    
    
    
    
    
    
    