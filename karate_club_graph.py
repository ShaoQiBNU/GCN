#################### load packages ##################
import networkx as nx
from networkx.algorithms.approximation import k_components
import matplotlib.pyplot as plt

#################### 导入网络图 ##################
G = nx.karate_club_graph()

######### 查看网络nodes ########
print(G.nodes())

######### 查看网络nodes的label，即每个node所属的类别 ########
def build_k_number_dict(G_k_components):
    k_components_dict = {}

    for k, comps in sorted(G_k_components.items()):
        for comp in comps:
            for node in comp:
                k_components_dict[node] = k-1

    return k_components_dict

G_k_components = k_components(G)
k_components_dict = build_k_number_dict(G_k_components)
print(k_components_dict)
print(list(k_components_dict.values()))

######### 查看网络图 ########
colors = ['red', 'green', 'blue', 'yellow']
color = []
for v in k_components_dict.values():
    color.append(colors[v-1])

nx.draw(G, with_labels=True, node_size=40, node_color=color)
plt.show()