
from warnings import deprecated

import networkx as nx
import matplotlib.pyplot as plt

@deprecated
def main():
    n_nodes = 4
    # 生成4顶点的3-regular图（K4）
    G = nx.complete_graph(n_nodes)
    edges = list(G.edges())

    # 绘制并保存图
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.savefig('./QAOA/graph.jpg')
    plt.close()

    # 暴力求解最大切割并记录所有最优解
    max_cut_value = 0
    best_cuts = []  # 存储所有最优解的二进制串

    for i in range(2**n_nodes):
        s = format(i, f'0{n_nodes}b')  # 转换为n位二进制字符串（如 '0011'）
        cut = 0
        for (u, v) in edges:
            if s[u] != s[v]:  # 若边的两端在不同集合
                cut += 1
        # 更新最大值和最优解列表
        if cut > max_cut_value:
            max_cut_value = cut
            best_cuts = [s]  # 发现更大的值，重置列表
        elif cut == max_cut_value:
            best_cuts.append(s)  # 同等最大值，添加到列表

    # 输出结果
    print(f"Optimal Max-Cut Value: {max_cut_value}")
    print("Optimal Bitstrings:")
    for bits in best_cuts:
        print(f"- {bits} (顶点分组: {[int(c) for c in bits]})")



    import pennylane as qml
    from pennylane import numpy as np
    from pennylane import qaoa

    # 定义cost_layer mix_layer 量子设备
    cost_h, mixer_h = qaoa.maxcut(G)
    dev = qml.device("default.mixed", wires=range(n_nodes))
    # 初始化参数
    p = 2  # 层数
    params = np.array([[0.5]*p, [0.5]*p], requires_grad=True)  # 形状 (2, p)

    # 定义QAOA电路（p层）—— manual
    def U_C(gamma, graph):
        for (u, v) in graph.edges():
            # qml.IsingZZ(gamma, wires=[u,v])
            # ZZ 相互作用分解为 CNOT + RZ
            qml.CNOT(wires=[u, v])
            qml.RZ(2 * gamma, wires=v)  # 系数 2 对应哈密顿量权重
            qml.CNOT(wires=[u, v])

    def U_B(beta, n_qubits):
        for w in range(n_qubits):
            qml.RX(2 * beta, wires=w)  # RX(2β) 对应时间演化

    @qml.qnode(dev)
    def circuit(params):
        gamma = params[0]  # 形状 (p,)
        beta = params[1]   # 形状 (p,)

        # 初始化叠加态
        for w in range(n_nodes):
            qml.Hadamard(wires=w)
        
        # 手动实现 QAOA 层
        for i in range(p):
            U_C(gamma[i], G)
            U_B(beta[i], n_nodes)
        
        return qml.probs(wires=range(n_nodes))

    # 定义QAOA电路（p层）—— pennylane api
    # def qaoa_circuit(params):
    #     # 参数分解为gamma和beta列表
    #     gamma = params[0]  # 形状 (p,)
    #     beta = params[1]   # 形状 (p,)
    #     p = len(gamma)     # 层数
        
    #     # 初始化所有量子比特为叠加态
    #     for w in range(n_nodes):
    #         qml.Hadamard(wires=w)
        
    #     # 交替应用p层Cost和Mixer操作
    #     for i in range(p):
    #         qml.qaoa.cost_layer(gamma[i], cost_h)  # 第i层Cost
    #         qml.qaoa.mixer_layer(beta[i], mixer_h) # 第i层Mixer
        
    #     return qml.probs(wires=range(n_nodes))  # 返回测量概率

    # # 创建量子节点
    # @qml.qnode(dev)
    # def circuit(params):
    #     return qaoa_circuit(params)

    # 计算期望值
    def expectation(params):
        probs = circuit(params)
        exp_value = 0
        for i, prob in enumerate(probs):
            s = format(i, f'0{n_nodes}b')
            cut = 0
            for (u, v) in edges:
                if s[u] != s[v]:
                    cut += 1
            exp_value += prob * cut
        return exp_value

    # 使用梯度下降优化
    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    steps = 50
    for i in range(steps):
        params, cost = opt.step_and_cost(lambda p: -expectation(p), params)
        if i % 10 == 0:
            print(f"Step {i}: Expectation = {-cost:.4f}")

    # 输出最终参数和近似比
    final_expectation = expectation(params)
    approx_ratio = final_expectation / max_cut_value
    print(f"Optimized Parameters (gamma, beta):\n{params}")
    print(f"Approximation Ratio (p={p}): {approx_ratio:.4f}")

    # 绘制量子电路
    fig, ax = qml.draw_mpl(circuit,show_all_wires=True)(params)
    fig.savefig(f"./QAOA/circuit_p{p}.jpg")
    plt.close()

    # 绘制概率分布
    probs = circuit(params)
    plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), [format(i, '04b') for i in range(len(probs))], rotation=90)
    plt.xlabel('Quantum State')
    plt.ylabel('Probability')
    plt.title('QAOA Probability Distribution')
    plt.tight_layout()
    plt.savefig(f'./QAOA/probability_distribution_{p}.jpg')
    plt.close()

if __name__ == '__main__':
    main()