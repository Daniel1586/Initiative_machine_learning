#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_c4.5.py
"""
import pandas as pd
from math import log2


# 读取样本数据,最后1列为类别
def load_data(file):
    dataset = pd.read_csv(file)

    return dataset


# 计算样本集信息熵
def calc_entropy(dataset):
    num_set = dataset.shape[0]
    num_labels = {}
    ncols = dataset.columns.tolist()
    label = dataset[ncols[-1]].tolist()

    # 计算类别个数
    for item in label:
        if item not in num_labels.keys():
            num_labels[item] = 1
        else:
            num_labels[item] += 1

    # 计算信息熵
    info_ent = 0.0
    for key in num_labels:
        prob = num_labels[key]/num_set
        info_ent -= prob*log2(prob)

    return info_ent


# 计算样本集特征的熵
def calc_attr_entropy(dataset, axis):
    num_set = dataset.shape[0]
    num_labels = {}
    label = dataset[axis].tolist()

    # 计算类别个数
    for item in label:
        if item not in num_labels.keys():
            num_labels[item] = 1
        else:
            num_labels[item] += 1

    # 计算信息熵
    info_ent = 0.0
    for key in num_labels:
        prob = num_labels[key]/num_set
        info_ent -= prob*log2(prob)

    return info_ent


# 根据特征和特征取值对样本集进行划分
def split_dataset(dataset, axis, value):
    cols = dataset.columns.tolist()         # 样本集的特征
    axis_attr = dataset[axis].tolist()      # 待选择特征的取值

    # 1.去掉待选择特征后的样本集
    rest_dataset = pd.concat([dataset[item] for item in cols if item != axis], axis=1)

    # 2.根据待选择特征取值,获得子集
    i = 0
    drop_idx = []  # 删除项的索引集
    for axis_val in axis_attr:
        if axis_val != value:
            drop_idx.append(i)
            i += 1
        else:
            i += 1
    new_dataset = rest_dataset.drop(rest_dataset.index[drop_idx])

    return new_dataset


# 计算样本集每个特征的信息增益率
def gen_info_gain_ratio(dataset):
    attr_num = dataset.shape[1] - 1     # 根节点特征数
    info_ent = calc_entropy(dataset)    # 根节点信息熵
    best_gain_ratio = 0.0
    best_attr = -1
    cols = dataset.columns.tolist()

    # 遍历样本集中的特征,计算信息增益率
    for i in range(attr_num):
        attr_ent = 0.0
        attr_val = set(dataset[cols[i]].tolist())               # 样本集特征的取值
        attr_intrinsic = calc_attr_entropy(data_set, cols[i])   # 样本关于特征的熵

        # 对特征按取值划分子集,并计算信息增益
        for value in attr_val:
            sub_dataset = split_dataset(dataset, cols[i], value)    # 特征划分的子集
            prob = sub_dataset.shape[0] / dataset.shape[0]          # 子集权重
            attr_ent += prob * calc_entropy(sub_dataset)
        info_gain = info_ent - attr_ent
        info_gain_ratio = info_gain/attr_intrinsic
        print("%-20s%-30f" % (cols[i], info_gain_ratio))

        # 保存信息增益最大的特征
        if info_gain_ratio > best_gain_ratio:
            best_gain_ratio = info_gain_ratio
            best_attr = cols[i]

    return best_attr, best_gain_ratio


# 样本特征集为空集,但类别标签不完全相同,划分为类别最多的类
def major_count(labellist):
    labelcnt = {}
    for vote in labellist:
        if vote not in labelcnt.keys():
            labelcnt[vote] = 0
        labelcnt[vote] += 1
    sortedlabelcnt = sorted(labelcnt.items(), key=lambda x: x[1])

    return sortedlabelcnt[0][0]


# 以递归方式生成c45_tree
def gen_c45_tree(dataset, dropcol):
    data_attrs = dataset.columns.tolist()[:-1]                      # 样本特征集合
    data_label = dataset[dataset.columns.tolist()[-1]].tolist()     # 样本类别集合

    # 1.若样本集中所有实例属于同一类Ck,则为单节点树,并将Ck作为该节点的类标记
    if data_label.count(data_label[0]) == len(data_label):
        return data_label[0]

    # 2.若样本特征集为空集,则为单节点树,并将数据集中实例数最大的类Ck作为该节点的类标记
    if len(dataset[0:1]) == 0:
        return major_count(data_label)

    # 3.根据c4.5算法选择特征(样本集按特征划分后信息增益率最大)
    print("特征集和类别: ", dataset.columns.tolist())
    best_attr, best_gain_ratio = gen_info_gain_ratio(dataset)
    print("Best attribute:", best_attr)

    # 4.按照最优特征迭代生成id3 tree
    new_tree = {best_attr: {}}
    attr_value = dataset[best_attr]
    uniq_value = set(attr_value)
    for value in uniq_value:
        attr_set = split_dataset(dataset, best_attr, value)
        attr_tre = gen_c45_tree(attr_set, best_attr)
        new_tree[best_attr][value] = attr_tre

    return new_tree


if __name__ == "__main__":
    filename = "dataset.csv"
    data_set = load_data(filename)
    drop_col = []
    c45_tree = gen_c45_tree(data_set, drop_col)
    print(c45_tree)
