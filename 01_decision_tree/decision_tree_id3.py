#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: decision_tree_id3.py
"""
from math import log2


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


if __name__ == "__main__":

    # 测试数据:0-3列为属性,4列为类别
    test_data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']]

    id3_tree = gen_id3_tree(test_data)
    printtree(tree = tree)
    classify(['(direct)','USA','yes',5],tree)


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