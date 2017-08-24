#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

iris_feature = u'花萼长度', u'花萼宽度',u'花瓣长度',u'花瓣宽度'

if __name__== "__main__":  # ==号 而不是 =  有一个冒号

   #准备好数据
    path = "..\\Regression\\iris.data" #单引号
    data  = pd.read_csv(path, header=None) # head=None
    x,y =  data[range(4)],data[4] #我的书写 data(range([4]))
    y = pd.Categorical(y).codes
    x = x[[0,1]]
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, train_size= 0.6)

    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr') #核心 关键
    clf.fit(x_train,y_train)

    print clf.predict(x_test)
    print y_test
    print accuracy_score(y_test, clf.predict(x_test)) # 准确率
    print clf.score(x_test,y_test) #clf.score(y_test, clf.predict(x_test)) # 准确率
    # print clf.decision_function(x_train)
    print clf.decision_function(x_test)

     # 画图
    x1_min, x2_min = x.min() #取x的最小值
    x1_max, x2_max = x.max() #取x的最大值

    mpl.rcParams['font.sans-serif'] = [u'SimHei']  #指定默认字体
    plt.title(u'鸢尾花SVM二特征分类', fontsize=18) #标题设置
    mpl.rcParams['axes.unicode_minus'] = False  #解决负号'-'显示为方块的问题

    plt.xlabel(iris_feature[0], fontsize=13)  #设置x轴刻度的标签名字
    plt.ylabel(iris_feature[1], fontsize=13)  #设置x轴刻度的标签名字

    plt.xlim(x1_min,x1_max)  #设置x轴刻度的取值范围
    plt.ylim(x2_min,x2_max)  #设置y轴刻度的取值范围

    plt.figure(facecolor='w')   # 面板颜色

    plt.grid(b=True, ls=':') #   plt.grid(b=True, ls=':') 网格线的类型
    plt.tight_layout(pad=1.5)
    plt.show()

