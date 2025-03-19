from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pennylane as qml

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)

nx.draw(graph, with_labels=True, pos=positions)
plt.show()

# retrieve the cost Hamiltonian as well as a recommended mixer Hamiltonian
cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=False)

print("Cost Hamiltonian", cost_h)
print("Mixer Hamiltonian", mixer_h)

# build the cost and mixer layers
def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

# employ a circuit consisting of two QAOA layers
wires = range(4)
depth = 2
def circuit(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1])

# use the PennyLane-Qulacs plugin to run the circuit on the Qulacs simulator
dev = qml.device("qulacs.simulator", wires=wires)
@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)

# optimize the cost function
optimizer = qml.GradientDescentOptimizer()
steps = 70
params = np.array([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)

# optimize the circuit
for i in range(steps):
    params = optimizer.step(cost_function, params)

print("\nOptimal Parameters")
print(params)

# redefine the full QAOA circuit with the optimal parameters, but this time we return the probabilities of measuring each bitstring
@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)


probs = probability_circuit(params[0], params[1])

# display a bar graph showing the probability of measuring each bitstring
plt.style.use("seaborn-v0_8")
plt.bar(range(2 ** len(wires)), probs)
plt.show()