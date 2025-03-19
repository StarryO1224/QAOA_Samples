import pennylane as qml

H = qml.Hamiltonian([1, 1, 0.5], 
                    [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

# print(H)

dev = qml.device("default.qubit", wires=(0,1,2,3))
t = 1
n = 2

@qml.qnode(dev)

def circuit():
    qml.ApproxTimeEvolution(H, t, n)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

fig, ax = qml.draw_mpl(circuit)()
fig.savefig("./test.png")