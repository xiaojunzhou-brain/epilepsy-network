# coding:utf-8
import pandas as pd
import numpy as np
import networkx as nx
from tools.utils import divide


def mouse():
    '''
    input: Excel file with mouse
    return: G1, G2,
    '''
    data = pd.read_excel('connectome/mouse.xlsx', header=0, index_col=0, sheet_name=None, engine='openpyxl')
    metadate = pd.read_excel('connectome/mouse_meta.xlsx', sheet_name='Voxel Count_295 Structures')

    m = metadate.loc[metadate['Represented in Linear Model Matrix'] == 'Yes']

    columns = []
    cortices = [[0, 0]]
    for region in m['Major Region'].unique():
        i = [columns.append(acronym.replace(' ', ''))
             for acronym in m.loc[m['Major Region'] == region, 'Acronym'].values]
        cortices.append([cortices[-1][-1], cortices[-1][-1] + len(i)])
    cortices.remove([0, 0])

    regions = m['Major Region'].unique()

    d = data['W_ipsi']
    p = data['PValue_ipsi']
    d = d[columns].reindex(columns)
    p = p[columns].reindex(columns)

    d = d.values
    p = p.values

    p[np.isnan(p)] = 1

    d[p > 0.01] = 0

    matrix = np.zeros_like(d)

    for i in [1e-4, 1e-2, 1]:
        matrix[d >= i] += 1

    graph = nx.from_numpy_array(matrix)

    G1, G2, n1, n2 = divide(matrix, cortices)
    return graph, matrix, G1, G2, n1, n2, cortices, regions


def random(n=100):
    random_graph = nx.erdos_renyi_graph(n, p=0.1, seed=42)
    while not nx.is_connected(random_graph):
        random_graph = nx.erdos_renyi_graph(n, p=0.1, seed=42)
    random_matrix = nx.to_numpy_array(random_graph)

    return random_graph, random_matrix
    # nx.draw(random_graph, with_labels=True, font_weight='bold')
    # plt.title("Random Graph (Erdős-Rényi)")
    # plt.show()


def small_world(n=100):
    small_world_graph = nx.watts_strogatz_graph(n, k=6, p=0.2, seed=42)
    while not nx.is_connected(small_world_graph):
        small_world_graph = nx.watts_strogatz_graph(n, k=6, p=0.2, seed=42)
    small_world_matrix = nx.to_numpy_array(small_world_graph)

    return small_world_graph, small_world_matrix
    # nx.draw(small_world_graph, with_labels=True, font_weight='bold')
    # plt.title("Small World Graph (Watts-Strogatz)")
    # plt.show()


def scale_free(n=100):
    scale_free_graph = nx.barabasi_albert_graph(n, m=2, seed=42)
    while not nx.is_connected(scale_free_graph):
        scale_free_graph = nx.barabasi_albert_graph(n, m=2, seed=42)
    scale_free_matrix = nx.to_numpy_array(scale_free_graph)

    return scale_free_graph, scale_free_matrix
    # nx.draw(scale_free_graph, with_labels=True, font_weight='bold')
    # plt.title("Scale-Free Graph (Barabási-Albert)")
    # plt.show()


class networkAnalysis:
    def __init__(self, G):
        self.G = G

    # mean shortest path length
    def MSPL(self):
        average_shortest_path_length = nx.average_shortest_path_length(self.G)
        # print("Average Shortest Path Length:", average_shortest_path_length)
        return average_shortest_path_length

    # average clustering coefficient
    def ACC(self):
        average_clustering_coefficient = nx.average_clustering(self.G)
        # print("Average Clustering Coefficient:", average_clustering_coefficient)
        return average_clustering_coefficient

    # degree distribution
    def DD(self):
        degree_histogram = nx.degree_histogram(self.G)
        # print("Degree Histogram:", degree_histogram)
        return degree_histogram

    def BC(self):
        betweenness_centrality = nx.betweenness_centrality(self.G)
        return betweenness_centrality

    def weight_MSPL(self):
        average_weighted_shortest_path_length = nx.average_shortest_path_length(self.G, weight='weight')
        return average_weighted_shortest_path_length

    def weight_ACC(self):
        def weighted_degree_clustering_coefficient(nodes, direction):
            clustering_coefficients = {}
            for node in nodes:
                if isinstance(self.G, nx.DiGraph):
                    neighbors = list(direction(node))
                else:
                    neighbors = list(self.G.neighbors(node))
                total_weight = 0
                if len(neighbors) > 1:
                    for neighbor in neighbors:
                        if isinstance(self.G, nx.DiGraph):
                            total_weight += direction[neighbor][node]['weight']
                        else:
                            total_weight += self.G[node][neighbor]['weight']
                    avg_weight = total_weight / len(neighbors)
                    max_possible_weight = max([self.G[node][neighbor]['weight'] for neighbor in neighbors])
                    clustering_coefficients[node] = avg_weight / max_possible_weight
                else:
                    clustering_coefficients[node] = 0  # 如果节点只有一个邻居，则集聚系数为0
            return clustering_coefficients

        clustering_coefficients = weighted_degree_clustering_coefficient(self.G.nodes(),
                                                                         self.G.predecessors if isinstance(self.G,
                                                                                                           nx.DiGraph) else self.G.neighbors)

        total_clustering_coefficient = sum(clustering_coefficients.values())
        num_nodes = len(self.G.nodes())
        if num_nodes == 0:
            return 0  # 避免除以0错误
        return total_clustering_coefficient / num_nodes


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 100
    a, b, c = 'random', 'small_world', 'scall_free'
    use = b
    if use == 'random':
        net_graph, net_matrix = random(n)
    elif use == 'small_world':
        net_graph, net_matrix = small_world(n)
    elif use == 'scall_free':
        net_graph, net_matrix = scale_free(n)
    else:
        print('invalid')

    pos = nx.circular_layout(net_graph)
    nx.draw_networkx_nodes(net_graph, pos, node_size=4, node_color='skyblue')
    # nx.draw_networkx_nodes(graph, pos, nodelist=index, node_color='red', node_size=1.5)
    nx.draw_networkx_edges(net_graph, pos, alpha=1)
    plt.show()

    plt.imshow(net_matrix)
    plt.show()

    net_mouse, matrix = mouse()
    nx.draw(net_mouse, with_labels=True, font_weight='bold')
    plt.show()
    plt.imshow(matrix)
    plt.show()
