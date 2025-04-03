import os
import matplotlib.pyplot as plt
import pennylane as qml

from pennylane import numpy as np
from pennylane import qaoa

from classical_maxcut import naive_max_cut
from graph_generator import generate_3_regular_graph

n_nodes = 4
base_path=f'./QAOA/plus_graph{n_nodes}/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
qt_device='default.mixed'
G, edges = generate_3_regular_graph(n_nodes, base_path)
max_cut_value = naive_max_cut(n_nodes, edges)


# 初始化参数
p = 1 # 层数
params = np.array([[0.5], [0.5]], requires_grad=True)
params_plus = np.array([[0.5]*n_nodes, [0.5]*n_nodes], requires_grad=True)
# 定义cost_layer mix_layer 量子设备
cost_h, mixer_h = qaoa.maxcut(G)
dev = qml.device(qt_device, wires=range(n_nodes))

# 定义QAOA电路（p层）—— manual
def U_C(gamma, graph):
    for (u, v) in graph.edges():
        # qml.IsingZZ(gamma, wires=[u,v])
        # RZZ(θ) = CNOT(I⊗RZ(θ))CNOT
        qml.CNOT(wires=[u, v])
        qml.RZ(2 * gamma, wires=v)  # 系数 2 对应哈密顿量权重
        qml.CNOT(wires=[u, v])

def U_B(beta, n_qubits):
    for w in range(n_qubits):
        qml.RX(2 * beta, wires=w)  # RX(2β) 对应时间演化

@qml.qnode(dev)
def circuit(params):
    gamma = params[0]
    beta = params[1]
    gamma_plus = params_plus[0]
    beta_plus = params_plus[1]

    # 初始化叠加态
    for w in range(n_nodes):
        qml.Hadamard(wires=w)
    
    # 手动实现 QAOA 层
    for _ in range(p):
        U_C(gamma[0], G)
        U_B(beta[0], n_nodes)

    # QAOA-plus problem-indenpendent layer and are independently parameterized.
    # 两比特门层（线性连接）
    for i in range(n_nodes - 1):
        qml.IsingZZ(gamma_plus[i], wires=[i, i + 1])
    # 混合层（独立参数）
    for wire in range(n_nodes - 1):
        qml.RX(beta_plus[wire], wires=wire)
    
    return qml.probs(wires=range(n_nodes))

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

# 使用梯度下降优化(fixed step)
opt = qml.GradientDescentOptimizer(stepsize=0.01)
steps = 50
for i in range(steps):
    params, cost = opt.step_and_cost(lambda p: -expectation(p), params)
    params_plus, cost = opt.step_and_cost(lambda p: -expectation(p), params_plus)
    if i % 10 == 0:
        print(f"Step {i}: Expectation = {-cost:.4f}")

# 输出最终参数和近似比
final_expectation = expectation(params)
approx_ratio = final_expectation / max_cut_value
print(f"Optimized Parameters (gamma, beta):\n{params}")
print(f"Optimized Parameters (gamma, beta)_plus:\n{params_plus}")
print(f"Approximation Ratio (p={p}): {approx_ratio:.4f}")

# 绘制量子电路
fig, ax = qml.draw_mpl(circuit,show_all_wires=True, decimals=2)(params)
fig.savefig(f"{base_path}circuit_p{p}.jpg")
plt.close()

# 绘制概率分布
probs = circuit(params)
plt.bar(range(len(probs)), probs)

# 添加近似比文本标注（显示在左上角）
text_x = 0  # 横向位置（相对坐标，0=左边缘，1=右边缘）
text_y = 1 # 纵向位置（相对坐标，0=下边缘，1=上边缘）
plt.text(text_x, text_y, 
         f"Approximation Ratio: {approx_ratio:.4f}", 
         transform=plt.gca().transAxes,  # 使用相对坐标系
         fontsize=6,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))  # 添加背景框

plt.xticks(range(len(probs)), [format(i, f'0{n_nodes}b') for i in range(len(probs))], rotation=90)
plt.xlabel('Quantum State')
plt.ylabel('Probability')
plt.title(f'QAOA Probability Distribution p={p}')
plt.tight_layout()
plt.savefig(f'{base_path}probability_distribution_{p}.jpg')
plt.close()