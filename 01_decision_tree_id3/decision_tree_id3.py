#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: decision_tree_id3.py
"""
import numpy as np
import pandas as pd
from math import log2

"""
# 统计数据最后1列(属性/类别)出现次数
# 返回结果为字典
def datacount(dataset):
    results = {}
    for row in dataset:
        r = row[len(row)-1]
        if r not in results:
            results[r] = 0
        results[r] += 1

    return results


# 信息熵计算
def entropy(dataset):
    results = datacount(dataset)
    ent = 0.0
    for r in results.keys():
        prob = float(results[r])/len(dataset)
        ent = ent - prob*log2(prob)

    return ent


# 对数据集某列属性拆分子集,并返回子集
def gen_set(dataset, col, attr_value):
    value = list(attr_value.keys())
    subset = []
    for i in range(len(value)):
        split_func = lambda row: row[col] == value[i]
        set_init = [row for row in dataset if split_func(row)]
        subset.append(set_init)

    return subset


# 定义节点的属性
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # col是待检验的判断条件所对应的列索引值
        self.value = value  # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results  # 保存的是针对当前分支的结果，它是一个字典
        self.tb = tb  # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb  # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点


# 基尼不纯度
# 随机放置的数据项出现于错误分类中的概率
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:  # 这个循环是否可以用（1-p1）替换？
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# 改进giniimpurity
def giniimpurity_2(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts.keys():
        p1 = float(counts[k1]) / total
        imp += p1 * (1 - p1)
    return imp


# 以递归方式生成id3_tree
def gen_id3_tree(dataset, ent=entropy):
    if len(dataset) == 0:
        return decisionnode()
    root_ent = ent(dataset)

    # 最优属性划分状态
    opt_gain = 0.0
    opt_crit = None
    opt_sets = None

    attr_num = len(dataset[0]) - 1  # 样本待划分数目
    for col in range(0, attr_num):
        # 1.当前列属性取值
        attr_value = {}
        for row in dataset:
            attr_value[row[col]] = 1

        # 2.根据当前列属性取值,拆分子集,并计算属性划分数据集后信息增益
        subsets = gen_set(dataset, col, attr_value)
        gain = 0.0
        for setnum in range(len(subsets)):
            subset = subsets[setnum]
            set_prop = float(len(subset)) / len(dataset)  # 子集样本占比
            set_ent = ent(subset)  # 子集信息熵
            gain += set_prop * set_ent
        gain = root_ent - gain

        # 3.记录信息增益最大的属性
        if gain > opt_gain:
            opt_gain = gain
            opt_crit = (col, attr_value)
            opt_sets = subsets

    # 按信息增益最大创建子分支
    if opt_gain > 0:
        trueBranch = gen_id3_tree(best_sets[0])  # 递归调用
        falseBranch = gen_id3_tree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


# 决策树的显示
def printtree(tree, indent=''):
    # 是否是叶节点
    if tree.results != None:
        print
        str(tree.results)
    else:
        # 打印判断条件
        print
        str(tree.col) + ":" + str(tree.value) + "? "
        # 打印分支
        print
        indent + "T->",
        printtree(tree.tb, indent + " ")
        print
        indent + "F->",
        printtree(tree.fb, indent + " ")


# 对新的观测数据进行分类
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)
"""


"""
# 决策树的剪枝
def prune(tree,mingain):
    # 如果分支不是叶节点，则对其进行剪枝
    if tree.tb.results == None:
        prune(tree.tb,mingain)
    if tree.fb.results == None:
        prune(tree.fb,mingain)
    # 如果两个子分支都是叶节点，判断是否能够合并
    if tree.tb.results !=None and tree.fb.results !=None:
        #构造合并后的数据集
        tb,fb = [],[]
        for v,c in tree.tb.results.items():
            tb+=[[v]]*c
        for v,c in tree.fb.results.items():
            fb+=[[v]]*c
        #检查熵的减少量
        delta = entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)
        if delta < mingain:
            # 合并分支
            tree.tb,tree.fb = None,None
            tree.results = uniquecounts(tb+fb)
# test
tree = buildtree(my_data,scoref = giniimpurity)
prune(tree,0.1)
printtree(tree)
"""


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
    new_dataset = rest_dataset.drop(drop_idx)
    new_dataset.reset_index(drop=True)

    return new_dataset


# 计算样本集每个特征的信息增益
def gen_info_gain(dataset):
    attr_num = dataset.shape[1] - 1     # 根节点特征数
    info_ent = calc_entropy(dataset)    # 根节点信息熵
    best_gain = 0.0
    best_attr = -1
    cols = dataset.columns.tolist()

    # 遍历样本集中的特征,计算信息增益
    for i in range(attr_num):
        attr_val = set(dataset[cols[i]].tolist())      # 样本集特征的取值
        attr_ent = 0.0

        # 对特征按取值划分子集,并计算信息增益
        for value in attr_val:
            sub_dataset = split_dataset(dataset, cols[i], value)    # 特征划分的子集
            prob = sub_dataset.shape[0] / dataset.shape[0]          # 子集权重
            attr_ent += prob * calc_entropy(sub_dataset)
        info_gain = info_ent - attr_ent
        print("%-20s%-30f" % (cols[i], info_gain))

        # 保存信息增益最大的特征
        if info_gain > best_gain:
            best_gain = info_gain
            best_attr = cols[i]

    return best_attr, best_gain


def majorityCnt(classList):
    '''
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]


# 以递归方式生成id3_tree
def gen_id3_tree(dataset, dropcol):
    data_attrs = dataset.columns.tolist()[:-1]                      # 样本特征集合
    data_label = dataset[dataset.columns.tolist()[-1]].tolist()     # 样本类别集合

    # 1.若样本集中所有实例属于同一类Ck,则为单节点树,并将Ck作为该节点的类标记
    if data_label.count(data_label[0]) == len(data_label):
        return data_label[0]

    # 2.若样本特征集为空集,则为单节点树,并将数据集中实例数最大的类Ck作为该节点的类标记
    if len(dataset[0:1]) == 0:
        return majorityCnt(data_label)

    # dataset.drop(dropCol, axis=1, inplace=True)

    # 3.根据id3算法选择特征(样本集按特征划分后信息增益最大)
    print("特征集和类别: ", dataset.columns.tolist())
    best_attr, best_gain = gen_info_gain(dataset)
    print("Best attribute:", best_attr)

    myTree = {bestFeature: {}}

    # del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    print(bestFeature)
    featValues = dataset[bestFeature]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeature][value] = createTree(splitDataSet(dataset, bestFeature, value), bestFeature)
    return myTree


if __name__ == "__main__":
    filename = "dataset.csv"
    data_set = load_data(filename)
    drop_col = []
    id3_tree = gen_id3_tree(data_set, drop_col)
    print(id3_tree)
