#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'


if __name__ == "__main__":
    path = '..\\Regression\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x, y = data[range(4)], data[4]
    y = pd.Categorical(y).codes
    #print y
    x = x[[0, 1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # 分类器
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    #clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    print y_train
    print y_train.ravel()
    clf.fit(x_train, y_train.ravel())

    # 准确率
    print clf.score(x_train, y_train)  # 精度
    print '训练集准确率：', accuracy_score(y_train, clf.predict(x_train))
    print clf.score(x_test, y_test)
    print '测试集准确率：', accuracy_score(y_test, clf.predict(x_test))

    # decision_function
    # print 'decision_function:\n', clf.decision_function(x_train)   # 训练样本到决策面的距离
    # print '\npredict:\n', clf.predict(x_train)  # 对训练样本进行预测

    # 画图
    x1_min, x2_min = x.min() #取x的最小值
    x1_max, x2_max = x.max() #取x的最大值

    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    # print 'grid_test = \n', grid_test
    # Z = clf.decision_function(grid_test)    # 样本到决策面的距离
    # print Z
    grid_hat = clf.predict(grid_test)       # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
     # cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark =  mpl.colors.ListedColormap(['g', 'r', 'b'])
     # cm_dark =  mpl.colors.ListedColormap(['g', 'r'])

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) # 区块颜色设置
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 训练集样本
    # plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本

    plt.figure(facecolor='w')   # 面板颜色

    plt.xlabel(iris_feature[0], fontsize=13)  #设置x轴刻度的标签名字
    plt.ylabel(iris_feature[1], fontsize=13)  #设置x轴刻度的标签名字

    plt.xlim(x1_min, x1_max)  #设置x轴刻度的取值范围
    plt.ylim(x2_min, x2_max)  #设置y轴刻度的取值范围

    plt.title(u'鸢尾花SVM二特征分类', fontsize=18) #标题设置

    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=1.5)
    plt.show()
