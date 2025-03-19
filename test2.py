import pennylane as qml 
import os
    
@qml.qnode(qml.device('lightning.qubit', wires=(0,1,2,3)))
def circuit(x, z):
    qml.QFT(wires=(0,1,2,3))
    qml.Toffoli(wires=(0,1,2))
    qml.CSWAP(wires=(0,2,3))
    qml.RX(x, wires=0)
    qml.CRZ(z, wires=(3,0))
    return qml.expval(qml.PauliZ(0))

qml.drawer.use_style("default")
fig, ax = qml.draw_mpl(circuit)(1.2345, 1.2345)
fig.savefig("./circuit.png")