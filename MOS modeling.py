# -*- coding: utf-8 -*-
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


def dealWithNB(X, y):
    nb = GaussianNB()
    nb.fit(X, y)
    return nb
	
    
data_train=pd.read_csv("miRNA_train/cancer_train.csv")
feature=[]  ###feature数据集
for i in data_train.columns:
    if (i!='label') & (i!='sampleID') & (i!='OS.time'):
        feature.append(i)
sample=['sampleID']
#sample2=['sample']
sample_train=data_train[sample]
label=['label']
x_train=data_train[feature] ####训练集的特征集
y_train=data_train[label] 
x_train=x_train.as_matrix()  ###将训练集的特征集转变为矩阵
y_train=y_train.as_matrix()   ######将训练集的lable标签转变为矩阵
sample_train=sample_train.as_matrix()

###测试模块


file_path='cancer/'
file_list=os.listdir(file_path)
for f in file_list:
    data_test=pd.read_csv(file_path+f)
    sample1=['sampleID']
    sample_test=data_test[sample1]
    x_test=data_test[feature]   ###测试集的特征集
    y_test=data_test[label]      ###测试集的lable标签
    x_test=x_test.as_matrix()     ######将测试集的特征集转变为矩阵
    y_test=y_test.as_matrix()     ###将测试集的lable标签转变为矩阵
    sample_test_OS=data_test['OS.time']

    NB=dealWithNB(x_train,y_train)    
    NB.fit(x_train,y_train)
    predictions_NB = NB.predict(x_test)
    probablity_NB= NB.predict_proba(x_test)
    fpr_NB,tpr_NB, threshold_NB = roc_curve(y_test,probablity_NB[:, 1])

    predict_lable1=pd.DataFrame(predictions_NB)
    probablity=pd.DataFrame(predictions_NB)
    result1=[sample_test,sample_test_OS,pd.DataFrame(probablity_NB)]
    result1_new=pd.concat(result1,axis=1)  ###axis=1,按照列合并，=0按照行合并
    result1_new.to_csv('simulation_probality/cancer/'+f+'.csv',index=False)
