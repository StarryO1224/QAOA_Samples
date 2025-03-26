def naive_max_cut(n_nodes, edges, print_results=True):
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

    if print_results:
         # 输出结果
        print(f"Optimal Max-Cut Value: {max_cut_value}")
        print("Optimal Bitstrings:")
        for bits in best_cuts:
            print(f"- {bits} (顶点分组: {[int(c) for c in bits]})")

    return max_cut_value

