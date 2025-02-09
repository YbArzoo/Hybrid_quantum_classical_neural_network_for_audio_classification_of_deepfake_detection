import pennylane as qml
import unitary  # ✅ Un-commented import
import embedding

dev = qml.device('default.qubit', wires=8)

# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])

def conv_layer2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])

def conv_layer3(U, params):
    U(params, wires=[0, 4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    if len(params) == 0:
        raise ValueError("params is empty, cannot perform pooling")
    
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])

def pooling_layer2(V, params):
    V(params, wires=[2, 0])
    V(params, wires=[6, 4])

def pooling_layer3(V, params):
    V(params, wires=[0, 4])

# QCNN Structure
def QCNN_structure(U, params, U_params):
    
    expected_params = 3 * U_params + 6  # Update this if your architecture requires more/less params
    if len(params) != expected_params:
        raise ValueError(f"Expected {expected_params} parameters, but got {len(params)}.\n"
                         f"Check where trained_params is coming from!")
    
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)
    conv_layer2(U, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)
    conv_layer3(U, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)

@qml.qnode(dev)


    
    
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy'):
    print(f"Debug: Received {len(params)} parameters in QCNN")  # ✅ Add debug statement
    
    if X.shape[0] != 256:
        raise ValueError(f"Expected input of 256 features, but got {X.shape[0]}")


    # ✅ Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # ✅ Quantum Convolutional Neural Network
    if U == 'U_SU4':
        QCNN_structure(unitary.U_SU4, params, U_params)
    else:
        print("Invalid Unitary Ansatz")
        return False

    # ✅ Output probabilities
    if cost_fn == 'mse':
        return qml.expval(qml.PauliZ(4))
    elif cost_fn == 'cross_entropy':
        return qml.probs(wires=4)

