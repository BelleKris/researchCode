# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:13:49 2018

@author: yzj
"""
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.metrics import roc_curve, auc 
from scipy import interp  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
from numpy.random import seed 
seed(1)   ###给神经网络定义一个随机种子，保证每次结果不变


def neural_network(x_train,y_train):
    #scaler = StandardScaler()
    #scaler.fit(x_train)
    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=5000)
    mlp.fit(x_train,y_train)
    return mlp  

def dealWithSVM(x_train,y_train):
    svc = svm.SVC(probability = True)
    svc.fit(x_train,y_train)
    return svc
    
def dealWithLR(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr
    
def dealWithNB(X, y):
    nb = GaussianNB()
    nb.fit(X, y)
    return nb
	
train_file=os.listdir("combine_train")    
test_file=os.listdir("combine_test")  
n=len(train_file)   
for j in range(n):
    data_train=pd.read_csv("combine_train/"+train_file[j])
    data_test=pd.read_csv("combine_test/"+test_file[j])
    cancer_name=train_file[j][0:11]
    feature=[]  ###feature数据集
    for i in data_train.columns:
        if (i!='label') & (i!='sampleID') & (i!='OS.time'):
            feature.append(i)
    sample=['sampleID']
    sample_train=data_train[sample]
    sample_test=data_test[sample]

    label=['label']
    x_train=data_train[feature] ####训练集的特征集
    y_train=data_train[label]  ##训练集的lable标签
    x_test=data_test[feature]   ###测试集的特征集
    y_test=data_test[label]      ###测试集的lable标签
    x_train=x_train.as_matrix()  ###将训练集的特征集转变为矩阵
    y_train=y_train.as_matrix()   ######将训练集的lable标签转变为矩阵
    x_test=x_test.as_matrix()     ######将测试集的特征集转变为矩阵
    y_test=y_test.as_matrix()     ###将测试集的lable标签转变为矩阵
    sample_train=sample_train.as_matrix()
#sample_test=sample_test.as_matrix()
    sample_test_OS=data_test['OS.time']

###(1) neural_network   
    mlp=neural_network(x_train,y_train)
    predictions_neural_network = mlp.predict(x_test)
    probablity_neural_network= mlp.predict_proba(x_test)
    fpr_neural_network,tpr_neural_network, threshold_neural_network = roc_curve(y_test,probablity_neural_network[:, 1])



###(2)LR
    LR=dealWithLR(x_train,y_train)    
    LR.fit(x_train,y_train)
    predictions_LR = LR.predict(x_test)
    probablity_LR= LR.predict_proba(x_test)
    fpr_LR,tpr_LR,threshold_LR = roc_curve(y_test,probablity_LR[:, 1])

###(3)NB
    NB=dealWithNB(x_train,y_train) 
    NB.fit(x_train,y_train)
    predictions_NB = NB.predict(x_test)
    probablity_NB= NB.predict_proba(x_test)
    fpr_NB,tpr_NB, threshold_NB = roc_curve(y_test,probablity_NB[:, 1])


###(4)SVM
    SVM=dealWithSVM(x_train,y_train)    
    SVM.fit(x_train,y_train)
    predictions_SVM= SVM.predict(x_test)
    probablity_SVM= SVM.predict_proba(x_test)
    fpr_SVM, tpr_SVM, threshold_SVM = roc_curve(y_test,probablity_SVM[:, 1]) 


    fpr=(fpr_neural_network,fpr_LR,fpr_NB,fpr_SVM)
    tpr=(tpr_neural_network,tpr_LR,tpr_NB,tpr_SVM)



    labels=['neural_network','LR','NB','SVM']    
    def roc_curve_(fpr,tpr,labels):  
        colorTable = ['blue','red','yellow','black']  
        plt.figure()    
        lw = 2    
        plt.figure(figsize=(10,8))  
        for k in range(len(fpr)):  
            roc_auc = auc(fpr[k],tpr[k]) ###计算auc的值  
            plt.plot(fpr[k],tpr[k],color=colorTable[k],linewidth=2,label='%s ROC curve (area = %0.4f)' %(labels[k],roc_auc))     
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')    
        plt.xlim([0.0, 1.0])    
        plt.ylim([0.0, 1.05])    
        plt.xlabel('False Positive Rate', fontsize=16)    
        plt.ylabel('True Positive Rate', fontsize=16)    
        plt.title(cancer_name+" ROC curve", fontsize=20)    
        plt.legend(loc="lower right")    
        plt.savefig("ROC/"+cancer_name+".jpg")
        plt.show() 
    roc_curve_(fpr,tpr,labels)

