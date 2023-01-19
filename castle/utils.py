import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd

def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G
# 有一个问题，如果添加 2 次， （1， 2）， （1， 2）， 只会用后一次更新已经存在的

# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(G, mean = 0, var = 1, SIZE = 10000, perturb = [], sigmoid = True):
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    # perturb 扰乱
    # 这里加的都是mean = 0, variance = 1的 noise
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean,var,SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            g.append(np.random.normal(0,1,SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]: # if there is an edge into this node
                if sigmoid:
                    # 有个问题，这里加的是 1 + np.exp(- 父节点)， 实际sigmoid = 1 / (1 + np.exp(- g[edge[0]]))
                    g[edge[1]] += 1/1+np.exp(-g[edge[0]])
                else:
                    # 正常是做平方   
                    g[edge[1]] +=np.square(g[edge[0]])
    # 换x,y轴
    g = np.swapaxes(g,0,1)
    return pd.DataFrame(g, columns = list(map(str, list_vertex)))
