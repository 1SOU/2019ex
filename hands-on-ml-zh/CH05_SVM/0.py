# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:16:51 2019

@author: Yisoul
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"bd")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"g^") #???
    plt.axis(axes)
    plt.grid(True,which='both') # 默认值就行，
    """matplotlin.pyplot.grid(b, which, axis, color, 
    linestyle, linewidth， **kwargs)"""
    plt.xlabel(r"$x_1$", fontsize=20) #显示函数表达式
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0) # rotation 旋转角度，，
    #,若不加 rotation=0，会你是几针选装90度，即垂直向上

def plot_predictions(clf,axes): # 绘制 模型预测等高线
    x0s= np.linspace(axes[0],axes[1],100)
    x1s= np.linspace(axes[2],axes[3],100)
    x0,x1= np.meshgrid(x0s,x1s)
    X=np.c_[x0.ravel(),x1.ravel()]
    y_pred= clf.predict(X).reshape(x0.shape)
    y_decision= clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)
    plt.contourf(x0,x1,y_decision,cmap=plt.cm.brg,alpha=0.1)
    
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)    
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])
#polynomial_svm_clf.fit(X, y)

#==============================================================================
# plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.show()
# 
#==============================================================================
#==============================================================================
# plt.figure(figsize=(11,5))
# plt.subplot(121)
# plot_predictions(polyd10r100c3_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=10, r=100, C=3$", fontsize=18)
# 
# plt.subplot(122)
# plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=10, r=100, C=5$", fontsize=18)
#==============================================================================
#==============================================================================
# plt.subplot(221)
# plot_predictions(poly_kernel_svm_clf,[-1.5,2.5,-1,1.5])
# plot_dataset(X,y,[-1.5,2.5,-1,1.5])
# plt.title(r"$d=3,r=1,C=5$",fontsize=18)
# 
# plt.subplot(222)
# plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=10, r=100, C=5$", fontsize=18)
# 
# plt.subplot(223)
# plot_predictions(polyd3r100_svm, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=3, r=100, C=5$", fontsize=18)
#==============================================================================

#==============================================================================
# plt.subplot(224)
# plot_predictions( polyd10r1_svm, [-1.5, 2.5, -1, 1.5])
# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=10, r=1, C=5$", fontsize=18)
#==============================================================================

def gaussian_rbf(x,landmark,gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark,axis=1)**2)
    #
def plot_rbf():
    X1d=np.linspace(-4,4,9).reshape(-1,1) # 一维特征数据
    # X2d= np.c_[X1d,X1d**2] # 添加二次项特征,
    #添加多项式 特征，线性可分。
    y= np.array([0,0,1,1,1,1,0,0]) 
    
    gamma=0.3
    x1s= np.linspace(-4.5,4.5,200).reshape(-1,1)
    #高斯径向基函数 生成近似特征
    x2s= gaussian_rbf(x1s,-2,gamma) # 关于地标点 -2 的近似特征
    x3s= gaussian_rbf(x1s,1,gamma) # 关于地标点 1 的近似特征
    Xk= np.c_[gaussian_rbf(X1d,-2,gamma),gaussian_rbf(X1d,1,gamma)]
    yk= np.array([0,0,1,1,1,1,1,0,0])
    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.grid()
    plt.axhline(y=0,color='k') # 画一条水平线，y=0  颜色为？？
    plt.scatter(x=[-2,1],y=[0,0],s=150,alpha=0.5,c="red") # 标记 地标点
    plt.plot(X1d[:,0][yk==0],np.zeros(4),"bs")
    plt.plot(X1d[:,0][yk==1],np.zeros(5),"g^")
    plt.plot(x1s,x2s,"g--") # 近似特征X2的曲线
    plt.plot(x1s,x3s,"b:") # 近似特征X3的曲线
    plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])# 设置y轴刻度
    plt.xlabel(r"$x_1$",fontsize=14)
    plt.annotate(r"$\mathbf{x}$", # 添加注释
                 xy=(X1d[3,0],0), # 要注释的点
                 xytext=(-0.5,0.20), # 注释文本的起始位置
                 ha="center",
                 arrowprops= dict(facecolor= 'black',shrink=0.1),
                 fontsize=18,)
    plt.text(-2,0.9,"$x_2$",ha="center",fontsize=20) # 添加文本，，不也相当于注释？
    plt.text(1,0.9, "$x_3$",ha="center",fontsize=20)
    plt.axis([-4.5,4.5,-0.1,1.1])
    
    plt.subplot(122)
    plt.grid(True,which='both')
    plt.axhline(y=0,color='k') # 水平线 horizontal
    plt.axvline(x=0,color='k') # 垂直线 vertical
    plt.plot(Xk[:,0][yk==0],Xk[:,1][yk==0],"bs")
    plt.plot(Xk[:,0][yk==1],Xk[:,1][yk==1],"g^")
    # 画出新的近似特征点
    plt.xlabel(r"$x_2$",fontsize=20)
    plt.ylabel(r"$x_3$",fontsize=20,rotation=0) # 不加rotation 会垂直向上
    plt.annotate(r"$\phi\left(\mathbf{x}\right)$",
                 xy=(Xk[3,0],Xk[3,1]),
                 xytext=(0.65,0.50),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=18)
    plt.plot([-0.1,1.1],[0.57,-0.1],"r--",linewidth=3)
    plt.axis([-0.1,1.1,-0.1,1.1])
    
    plt.subplots_adjust(right=1)
    plt.show()
    
#==============================================================================
# rbf_kernel_svm_clf=Pipeline([
#         ("scaler",StandardScaler()),
#         ("svm_clf",SVC(kernel="rbf",gamma=5,C=0.001))
#         ])
# rbf_kernel_svm_clf.fit(X,y)
#==============================================================================

def rbf_svm():
    gamma1,gamma2= 0.1,5
    C1,C2= 0.001,1000
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
    svm_clfs=[]
    
    for gamma,C in hyperparams:
        rbf_kernel_svm_clf= Pipeline([
                ("scaler",StandardScaler()),
                ("svm_clf",SVC(kernel="rbf",gamma=gamma,C=C))
                ])
        rbf_kernel_svm_clf.fit(X,y)
        svm_clfs.append(rbf_kernel_svm_clf)
    
    plt.figure(figsize=(11,9))
    for i,svm_clf in enumerate(svm_clfs):
        plt.subplot(221+i) # 221+0 222 223 224
        plot_predictions(svm_clf,[-1.5,2.5,-1,1.5])
        plot_dataset(X,y,[-1.5,2.5,-1,1.5])
        gamma,C= hyperparams[i]
        plt.title(r"$\gamma={},C={}$".format(gamma,C),fontsize=16)
    plt.subplots_adjust(hspace=0.31)    #调整上下行间距
    """plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 四周间距
    wspace=None, hspace=None)列间距，行间距"""
    plt.show()


"""Regresssion"""
from sklearn.svm import LinearSVR
# LinearSVR 速度较快，但是不支持核技巧
def regression():
    np.random.seed(42)
    m=50
    X=2*np.random.rand(m,1)
    y=(4+3*X+np.random.randn(m,1)).ravel()
    
    svm_reg= LinearSVR(epsilon=1.5,random_state=42)
    # epsilon 是间隔，，手动设置？？？？
    svm_reg.fit(X,y)
    # 与svm_reg 一样？？
    svm_reg1= LinearSVR(epsilon=1.5,random_state=42)
    svm_reg2= LinearSVR(epsilon=0.5,random_state=42)
    svm_reg1.fit(X,y)
    svm_reg2.fit(X,y)
    
    def find_support_vectors(svm_reg,X,y):
        y_pred= svm_reg.predict(X)
        off_margin= (np.abs(y-y_pred) >= svm_reg.epsilon) # 找到与决策线距离大于“间隔”的点,返回布尔数组
        return np.argwhere(off_margin)  # 返回True的索引，即“间隔之外的点”
    svm_reg1.support_=find_support_vectors(svm_reg1,X,y)
    svm_reg2.support_=find_support_vectors(svm_reg2,X,y)
    # .support_  是自定义的属性吗？
    
    eps_x1=1
    eps_y_pred=svm_reg1.predict([[eps_x1]])
    
    def plot_svm_regression(svm_reg,X,y,axes):
        x1s= np.linspace(axes[0],axes[1],100).reshape(100,1)
        y_pred= svm_reg.predict(x1s)
        plt.plot(x1s,y_pred,"k-",linewidth=2,label=r"$\hat{y}$") 
        # label的内容会显示在"图例"，由legend设置
        plt.plot(x1s,y_pred+svm_reg.epsilon,"k--")
        plt.plot(x1s,y_pred-svm_reg.epsilon,"k--")# 虚线 画出间隔
        # 不过这个间隔是用 垂直距离？？？不是应该是几何距离吗？
        plt.scatter(X[svm_reg.support_],y[svm_reg.support_],s=180,facecolors='#FFAAAA')
        plt.plot(X,y,"bo")
        plt.xlabel(r"$x_1$",fontsize=18)
        plt.legend(loc="upper left",fontsize=18)
        plt.axis(axes)
    
    plt.figure(figsize=(9,4))
    plt.subplot(121)
    plot_svm_regression(svm_reg1,X,y,[0,2,3,11])
    plt.title(r"$\epsilon={}$".format(svm_reg1.epsilon),fontsize=18)
    plt.ylabel(r"$y$",rotation=0)
    plt.annotate( # 带指向性的标注的注释 plt.text是纯文本标注
                 '',xy=(eps_x1,eps_y_pred),
                xytext=(eps_x1,eps_y_pred-svm_reg1.epsilon),
       arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
            )
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    plt.subplot(122)
    plot_svm_regression(svm_reg2,X,y,[0,2,3,11])
    plt.title(r"$\epsilon={}$".format(svm_reg2.epsilon),fontsize=18)
    plt.show()


from sklearn.svm import SVR
# 可以使用核技巧
def regre_poly():
    np.random.seed(42)
    m=100
    X=2* np.random.rand(m,1)-1
    y=(0.2 + 0.1*X + 0.5*X**2 + np.random.randn(m,1)/10).ravel() # 生成的数据点是 二次型的
    # ，，所以如果预测模型也是二次型，将会十分拟合

    svm_poly_reg1= SVR(kernel="poly",degree=2,C=100,epsilon=0.1) # 没有设置gamma ，此次不用对比
    svm_poly_reg1.fit(X,y)
    svm_poly_reg2= SVR(kernel="poly",degree=2,C=0.01,epsilon=0.1)
    svm_poly_reg2.fit(X,y)

    def plot_svm_regression(svm_reg,X,y,axes): #
        x1s= np.linspace(axes[0],axes[1],100).reshape(100,1)
        y_pred= svm_reg.predict(x1s)
        plt.plot(x1s,y_pred,"k-",linewidth=2,label=r"$\hat{y}$")
        # label的内容会显示在"图例"，由legend设置
        plt.plot(x1s,y_pred+svm_reg.epsilon,"k--")
        plt.plot(x1s,y_pred-svm_reg.epsilon,"k--")# 虚线 画出间隔
        # 不过这个间隔是用 垂直距离？？？不是应该是几何距离吗？
        plt.scatter(X[svm_reg.support_],y[svm_reg.support_],s=180,facecolors='#FFAAAA')
        plt.plot(X,y,"bo")
        plt.xlabel(r"$x_1$",fontsize=18)
        plt.legend(loc="upper left",fontsize=18)
        plt.axis(axes)

    plt.figure(figsize=(9,4))
    plt.subplot(121)
    plot_svm_regression(svm_poly_reg1,X,y,[-1,1,0,1])
    plt.title(r"$degree={},C={},\epsilon={}$".format(svm_poly_reg1.degree,
                                                     svm_poly_reg1.C,svm_poly_reg1.epsilon))
    plt.ylabel(r"y",fontsize=18,rotation=0)
    plt.subplot(122)
    plot_svm_regression(svm_poly_reg2,X,y,[-1,1,0,1])
    plt.title(r"$degree={},C={},\epsilon={}$".format(svm_poly_reg2.degree,
                                                     svm_poly_reg2.C, svm_poly_reg2.epsilon))
    plt.show()

regre_poly()

 
    
    
    
    
    
    
    
    
    
    