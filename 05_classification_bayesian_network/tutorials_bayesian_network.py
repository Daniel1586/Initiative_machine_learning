#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_bayesian_network.py
"""
import numpy as np


# 生成贝叶斯网络有向无环图和条件概率表
def gen_dataset():
    bayes_net = {}
    node_attr = ['Smoker', 'Coal_miner', 'Lung_cancer', 'Emphysema']
    node_dags = np.array([[0, 0, 1, 1],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
    print('代码待后续完善')
    node_cpts = 0
    bayes_net['node'] = node_attr
    bayes_net['dags'] = node_dags
    bayes_net['cpts'] = node_cpts

    return bayes_net


if __name__ == "__main__":
    data_set = gen_dataset()
