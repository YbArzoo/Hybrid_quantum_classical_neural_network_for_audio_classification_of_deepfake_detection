import pennylane as qml
# import unitary
import embedding

dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy'):


    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)
