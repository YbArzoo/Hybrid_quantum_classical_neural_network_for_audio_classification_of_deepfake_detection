from pennylane import numpy as np
import data  # ✅ Correct import


import qcnn_training  # ✅ Import QCNN training module

# Circuit training parameters
steps = 200
learning_rate = 0.01
batch_size = 25

# Load dataset
X_train, X_test, Y_train, Y_test = data.data_load_and_process(r"D:\BRACU\THESIS\CODE\spectrograms_images")




# Quantum unitary ansatz
U = 'U_SU4'
U_params = 15
embedding_type = "Amplitude"

# ✅ Correct function call (NO circular import)
from qcnn_training import circuit_training  # ✅ Import from qcnn_training.py

# Train the QCNN model
loss_history, trained_params = circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit='QCNN', cost_fn='cross_entropy')

# Save trained QCNN model
np.save("trained_qcnn_params.npy", trained_params)
print("✅ QCNN Training Completed!")
