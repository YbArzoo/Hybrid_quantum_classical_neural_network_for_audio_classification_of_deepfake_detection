# Implementation of Quantum circuit training procedure
# import QCNN_circuit
# import Hierarchical_circuit
from pennylane import numpy as np
import Training

# Circuit training parameters
steps = 200
learning_rate = 0.01
batch_size = 25

#Quantum unitary ansatz
U = 'U_SU4'
U_params = 15
embedding_type = "Amplitude"

#Train the QCNN model
loss_history, trained_params = Training.circuit_training(X_train, Y_train, U_params, embedding_type, circuit='QCNN', cost_fn='cross_entropy')
