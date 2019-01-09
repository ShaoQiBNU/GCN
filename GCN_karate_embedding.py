#################### load packages ##################
import networkx as nx
from networkx.algorithms.approximation import k_components
import matplotlib.pyplot as plt
from networkx import to_numpy_matrix
import numpy as np
import tensorflow as tf


#################### 导入网络图 ##################
G = nx.karate_club_graph()

######### 获取网络nodes的label，即每个node所属的类别 ########
def build_k_number_dict(G_k_components):
    k_components_dict = {}

    for k, comps in sorted(G_k_components.items()):
        for comp in comps:
            for node in comp:
                k_components_dict[node] = k

    return k_components_dict

G_k_components = k_components(G)
k_components_dict = build_k_number_dict(G_k_components)

######### 节点的颜色设置 ########
colors = ['red', 'green', 'blue', 'yellow']
color = []
for v in k_components_dict.values():
    color.append(colors[v-1])


#################### 获取网络图的输入矩阵和节点的特征矩阵 ####################
########### 网络图的邻接矩阵 ###########
adj = nx.adj_matrix(G) # 也可以用这种方式 A = to_numpy_matrix(G, nodelist=sorted(list(G.nodes())))
nodes = adj.shape[0]

########### 网络图的闭环矩阵=邻接矩阵+自身闭环矩阵 ###########
adj_tilde = adj + np.identity(n=adj.shape[0])

########### 网络图的节点度矩阵及其-1/2逆矩阵 ###########
d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)

########### 网络图的输入矩阵 ###########
adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)

########### 节点的特征矩阵 由于该网络每个节点没有特征，所以采用其自身的闭环矩阵作为特征矩阵 ###########
feat_x = np.identity(n=adj.shape[0])


#################### GCN结构 ####################
'''
def GCN_embedding(adj_norm, x):

    fc1 = tf.layers.dense(tf.matmul(adj_norm, x), 4)
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.layers.dense(tf.matmul(adj_norm, fc1), 4)
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.layers.dense(tf.matmul(adj_norm, fc2), 2)
    fc3 = tf.nn.relu(fc3)

    return fc3
'''

def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

weights = {
    'wc1': weight_var('wc1', [34, 4]),

    'wc2': weight_var('wc2', [4, 4]),

    'wc3': weight_var('wc3', [4, 2]),
}

def GCN_embedding(adj_norm, x):

    fc1 = tf.matmul(tf.matmul(adj_norm, x), weights['wc1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.matmul(tf.matmul(adj_norm, fc1), weights['wc2'])
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.matmul(tf.matmul(adj_norm, fc2), weights['wc3'])
    fc3 = tf.nn.relu(fc3)

    return fc3

out = GCN_embedding(adj_norm, feat_x)

##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    out = sess.run(out)

    for i in range(len(out)):
        plt.scatter(out[i][0], out[i][1], color=color[i])
    plt.show()