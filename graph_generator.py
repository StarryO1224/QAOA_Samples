import networkx as nx
import matplotlib.pyplot as plt

def generate_3_regular_graph(n):
    """
    生成一个随机的3-regular无向图。
    
    参数:
    n (int): 顶点数，必须是偶数且至少为4。
    
    返回:
    networkx.Graph: 生成的3-regular图实例。
    
    异常:
    ValueError: 如果顶点数不满足条件。
    """
    if n % 2 != 0:
        raise ValueError("顶点数必须是偶数。")
    if n < 4:
        raise ValueError("顶点数至少为4。")
    G = nx.random_regular_graph(3, n)
    return G

def plot_graph(G, title="3-Regular Graph"):
    """
    绘制图的拓扑结构。
    
    参数:
    G (networkx.Graph): 要绘制的图。
    title (str): 图的标题。
    """
    # 设置绘图布局和样式
    pos = nx.spring_layout(G)  # 使用Spring布局算法让节点分布更均匀
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color="skyblue",
        node_size=500,
        edge_color="gray",
        font_size=10,
    )
    plt.title(title)
    plt.axis("off")  # 隐藏坐标轴
    plt.show()

# 示例用法
if __name__ == "__main__":
    try:
        n = 20  # 修改顶点数测试（如8, 10等）
        graph = generate_3_regular_graph(n)
        print(f"生成的图有 {graph.number_of_nodes()} 个顶点和 {graph.number_of_edges()} 条边")
        print("边列表:", graph.edges())
        print("顶点度数:", dict(graph.degree()))
        
        for edge in graph.edges():
            print(edge[0], edge[1])
        # 绘制图形
        # plot_graph(graph, title=f"3-Regular Graph with {n} Nodes")
        
    except ValueError as e:
        print(e)