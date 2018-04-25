#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_knn.py
"""
import numpy as np
import pandas as pd


# 读取样本数据,最后1列为类别
def load_data(file):
    dataset = pd.read_csv(file)

    return dataset


# 数据集划分训练集和测试集(当前设置测试集仅1个数据)
# 返回数据类型为n维数组格式
def split_dataset(dataset, test_size=1):
    cols = dataset.columns.tolist()
    data_idx = dataset.index.tolist()
    test_idx = data_idx[-test_size:]
    train_idx = [i for i in data_idx if i not in test_idx]

    # 计算训练集, 转化为n维数组格式
    trainset = dataset.iloc[train_idx]
    train_datas = trainset.values[:, 0:len(cols)-1]
    train_label = trainset.values[:, -1:]

    # 计算测试集, 转化为n维数组格式
    testset = dataset.iloc[test_idx]
    test_datas = testset.values[:, 0:len(cols)-1]
    test_label = testset.values[:, -1:]

    return train_datas, train_label, test_datas, test_label


# 分类算法--朴素贝叶斯
class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.x = None
        self.y = None
        self.paras = []    # 存储数据集中每个特征中每个特征值出现概率

    # 计算不同类别不同特征不同特征值的条件概率
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(y)

        # 遍历所有类别
        for i in range(len(self.classes)):
            c = self.classes[i]
            c_idx = np.where(y == c)
            c_set = x[c_idx[0]]
            self.paras.append([])

            # 遍历相同类别不同特征
            for j in range(c_set.shape[1]):
                c_para = {}
                c_attr = np.unique(c_set[:, j])

                # 遍历相同特征不同特征值
                for attr_value in c_attr:
                    attr_value_num = c_set[c_set[:, j] == attr_value].shape[0]
                    attr_value_pro = attr_value_num/c_set.shape[0]
                    c_para[attr_value] = attr_value_pro
                self.paras[i].append(c_para)

    # 计算类别先验概率
    def calc_prior_prob(self, c):
        c_idx = np.where(self.y == c)
        c_set = self.x[c_idx[0]]
        prior = c_set.shape[0]/self.x.shape[0]

        return prior

    # 单个样本数据分类
    def classify(self, sample):
        posteriors = []

        # 遍历所有类别,计算后验概率
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calc_prior_prob(c)
            posterior = prior

            # 遍历所有特征的特征值
            for j, params in enumerate(self.paras[i]):
                sample_attr = sample[j]     # 预测样本第j个特征的值
                prob = params.get(sample_attr)
                posterior *= prob

            posteriors.append(posterior)

        # 后验概率排序,找出最大后验概率
        idx_of_max = np.argmax(posteriors)
        max_value = posteriors[idx_of_max]
        max_class = self.classes[idx_of_max]
        print("The max posterior prob: ", max_value)
        print("Classes by Naive Bayes: ", max_class)

        return max_class

    # 对数据集进行类别预测
    def predict(self, x):
        y_predict = []
        for sample in x:
            y = self.classify(sample)
            y_predict.append(y)

        return np.array(y_predict)


if __name__ == "__main__":
    filename = "dataset.csv"
    data_set = load_data(filename)
    x_train, y_train, x_test, y_test = split_dataset(data_set, test_size=1)

    clf = NaiveBayes()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
