import os
from matplotlib import pyplot as plt
import pennylane as qml
import decimal
decimal.getcontext().prec = 30
from pennylane import numpy as np
from scipy.optimize import minimize

from classical_maxcut import naive_max_cut
from graph_generator import generate_3_regular_graph

# 定义量子设备和问题参数
n_qubits = 4
base_path=f'./QAOA/plus_graph{n_qubits}/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
G, edges = generate_3_regular_graph(n_qubits, base_path)
c_max = naive_max_cut(n_qubits, edges)
p_depth = 1 

# 初始化量子设备
dev = qml.device("default.mixed", wires=n_qubits)

# 定义QAOA+量子电路
def qaoa_plus_circuit(params, draw_circuit=False):
    # 参数分割：前2p个参数为传统层（gamma和beta），剩余为问题无关层
    params_qa = params[:2 * p_depth].reshape(p_depth, 2)  # 传统QAOA参数
    gamma2_list = params[2 * p_depth: 2 * p_depth + (n_qubits - 1)]  # 问题无关层gamma
    beta2_list = params[2 * p_depth + (n_qubits - 1):]  # 问题无关层beta
    
    # 初始化为|+>态
    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)
    
    # p-QAOA层
    for layer in range(p_depth): 
        gamma, beta = params_qa[layer]
        # 相位分离层
        for u, v in edges:
            # qml.IsingZZ(2 * gamma, wires=[u, v])
            qml.CNOT(wires=[u, v])
            qml.RZ(2 * gamma, wires=v)
            qml.CNOT(wires=[u, v])
        # 混合层
        for wire in range(n_qubits):
            qml.RX(2 * beta, wires=wire)
        
    # QAOA+的问题无关层
    # 两比特门层（线性连接）
    for i in range(n_qubits - 1):
        qml.IsingZZ(2 * gamma2_list[i], wires=[i, i + 1])
    # 混合层（独立参数）
    for wire in range(n_qubits):
        qml.RX(2 * beta2_list[wire], wires=wire)
    
    # 构造完整的哈密顿量（包含常数项）
    coeffs = [-0.5] * len(edges) + [0.5 * len(edges)]
    obs = [qml.PauliZ(u) @ qml.PauliZ(v) for (u, v) in edges] + [qml.Identity(0)]
    H = qml.Hamiltonian(coeffs, obs)

    # 绘制电路图
    if draw_circuit:
        fig, ax = qml.draw_mpl(qaoa_plus_circuit, show_all_wires=True, decimals=2)(params)
        plt.savefig(f'{base_path}/circuit_{p_depth}.jpg', dpi=300, bbox_inches='tight')

    return qml.expval(H), qml.probs(wires=range(n_qubits))

# 创建QNode（使用自动微分接口）
qaoa_plus = qml.QNode(qaoa_plus_circuit, dev, interface="autograd")

# 定义损失函数（最大化期望值等价于最小化负期望）
def cost(params):
    expval, _ = qaoa_plus(params)
    return -expval

# 多次随机初始化优化（示例为3次）
best_ar = 0
best_params = None
num_trials = 1
total_params = 2 * p_depth + (n_qubits - 1) + n_qubits  # 参数总数 = 2p + (n-1) + n

for trial in range(num_trials):
    # 随机初始化参数（范围[0, π]）
    params = np.random.uniform(0, np.pi, size=total_params)
    
    # 使用BFGS优化
    result = minimize(cost, params, method='BFGS', options={'maxiter': 50})
    optimized_params = result.x
    
    # 计算近似比
    final_energy,_ = qaoa_plus(optimized_params)
    print(f'final_energy: {final_energy}')
    ar = decimal.Decimal(final_energy) / decimal.Decimal(c_max)
    
    if ar > best_ar:
        best_ar = ar
        best_params = optimized_params

    print(f"Trial {trial+1}: Approximation Ratio = {ar:.20f}")

_, best_probs = qaoa_plus(best_params, draw_circuit=True)  # 触发绘图功能
print(f"\nBest Approximation Ratio: {best_ar:.20f}")

# 将概率与对应的二进制字符串关联
bitstrings = [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]
prob_dict = {bit: prob for bit, prob in zip(bitstrings, best_probs)}

# 筛选概率大于1%的状态
filtered_probs = {k: v for k, v in prob_dict.items() if v > 0.01}

# 可视化概率分布
plt.figure(figsize=(10, 5))
plt.bar(filtered_probs.keys(), filtered_probs.values())
plt.xticks(rotation=45)
plt.xlabel("Bitstrings")
plt.ylabel("Probability")
plt.title(f"QAOA+ Final State Probabilities (p={p_depth})")
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{base_path}/prob_depth_{p_depth}.jpg", dpi=300)