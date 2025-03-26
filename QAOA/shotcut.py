params = np.array([[0.5]*p, [0.5]*p], requires_grad=True)
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
    if i % 10 == 0:
        print(f"Step {i}: Expectation = {-cost:.4f}")