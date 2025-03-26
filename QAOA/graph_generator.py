import networkx as nx
import matplotlib.pyplot as plt

def generate_3_regular_graph(n_nodes, img_save_path, is_save_img=True):
    # 生成N顶点的3-regular图
    G = nx.complete_graph(n_nodes)
    edges = list(G.edges())

    # 绘制并保存图
    nx.draw(G, with_labels=True, node_color='#66CCFF', node_size=800)
    plt.savefig(f'{img_save_path}graph.jpg')
    plt.close()

    return G, edges