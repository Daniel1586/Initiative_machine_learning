#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_cart.py
"""
import pandas as pd
from math import log2
from copy import deepcopy


# 读取样本数据,最后1列为类别
def load_data(file):
    dataset = pd.read_csv(file)

    return dataset


# 计算样本集基尼指数
def calc_gini(dataset):
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

    # 计算基尼指数
    gini_idx = 1.0
    for key in num_labels:
        prob = num_labels[key]/num_set
        gini_idx -= prob*prob

    return gini_idx


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


# 根据特征和特征取值对样本集进行划分,二分类
def split_dataset(dataset, axis, value):
    cols = dataset.columns.tolist()         # 样本集的特征
    axis_attr = dataset[axis].tolist()      # 待选择特征的取值

    # 1.去掉待选择特征后的样本集
    rest_dataset = pd.concat([dataset[item] for item in cols if item != axis], axis=1)

    # 2.根据待选择特征取值,获得子集
    i = 0
    drop_idx1 = []          # 等于value的索引值
    drop_idx2 = []          # 不等于value的索引值
    for axis_val in axis_attr:
        if axis_val != value:
            drop_idx2.append(i)
            i += 1
        else:
            drop_idx1.append(i)
            i += 1
    new_dataset1 = rest_dataset.drop(rest_dataset.index[drop_idx2])
    new_dataset2 = rest_dataset.drop(rest_dataset.index[drop_idx1])

    return new_dataset1, new_dataset2


# 计算样本集每个特征的基尼指数
def gen_gini_index(dataset):
    attr_num = dataset.shape[1] - 1     # 根节点特征数
    best_gini_index = 1.0
    best_attr = -1
    cols = dataset.columns.tolist()

    # 遍历样本集中的特征,计算基尼指数
    for i in range(attr_num):
        attr_val = set(dataset[cols[i]].tolist())               # 样本集特征的取值

        # 对特征按取值划分子集(二分类),并计算基尼指数
        for value in attr_val:
            attr_set = deepcopy(attr_val)
            sub_dataset1, sub_dataset2 = split_dataset(dataset, cols[i], value)     # 特征划分的子集
            prob1 = sub_dataset1.shape[0] / dataset.shape[0]                        # 子集权重
            prob2 = sub_dataset2.shape[0] / dataset.shape[0]
            attr_gini = prob1 * calc_gini(sub_dataset1) + prob2 * calc_gini(sub_dataset2)
            attr_set.remove(value)
            attr_str = init_str.join(attr_set)

            # 保存基尼指数最小的特征划分
            if attr_gini < best_gini_index:
                best_gini_index = attr_gini
                best_attr = [cols[i], value, attr_str]

            print("%-20s%-20s%-20s%-30f" % (cols[i], value, attr_str, attr_gini))

    return best_attr, best_gini_index


# 样本特征集为空集,但类别标签不完全相同,划分为类别最多的类
def major_count(labellist):
    labelcnt = {}
    for vote in labellist:
        if vote not in labelcnt.keys():
            labelcnt[vote] = 0
        labelcnt[vote] += 1
    sortedlabelcnt = sorted(labelcnt.items(), key=lambda x: x[1])

    return sortedlabelcnt[0][0]


# 以递归方式生成cart_tree
def gen_cart_tree(dataset, dropcol):
    data_attrs = dataset.columns.tolist()[:-1]                      # 样本特征集合
    data_label = dataset[dataset.columns.tolist()[-1]].tolist()     # 样本类别集合

    # 1.若样本集中所有实例属于同一类Ck,则为单节点树,并将Ck作为该节点的类标记
    if data_label.count(data_label[0]) == len(data_label):
        return data_label[0]

    # 2.若样本特征集为空集,则为单节点树,并将数据集中实例数最大的类Ck作为该节点的类标记
    if len(dataset[0:1]) == 0:
        return major_count(data_label)

    # 3.根据cart算法选择特征(样本集按特征划分后基尼指数最小)
    print("特征集和类别: ", dataset.columns.tolist())
    best_attr, best_gini = gen_gini_index(dataset)
    print("Best attribute:", best_attr)

    # 4.按照最优特征迭代生成cart tree
    new_tree = {best_attr[0]: {}}
    attr_set = split_dataset(dataset, best_attr[0], best_attr[1])
    for i in range(2):
        attr_tre = gen_cart_tree(attr_set[i], best_attr[0])
        new_tree[best_attr[0]][best_attr[i+1]] = attr_tre

    return new_tree


if __name__ == "__main__":
    filename = "dataset.csv"
    data_set = load_data(filename)
    drop_col = []
    cart_tree = gen_cart_tree(data_set, drop_col)
    print(cart_tree)
