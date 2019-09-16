# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:56:16 2019

@author: Yisoul
"""

from sklearn.datasets import fetch_mldata
import numpy as np
mnist= fetch_mldata('MNIST original',data_home= './dataset/')
x,y= mnist['data'],mnist['target']
shuffle_index= np.random.permutation(60000)
x_train,x_test,y_train,y_test= x[:60000],x[60000:],y[:60000],y[60000:]
x_train,y_train= x_train[shuffle_index],y_train[shuffle_index] 
y_train5= (y_train == 5)
y_test5= (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf= SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train5)

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf,x_train,y_train5,cv=3)
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
y_scores= cross_val_predict(sgd_clf,x_train,y_train5,cv=3,method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train5,y_scores[:,1])