import QCNN_circuit
import numpy as np
from sklearn.metrics import accuracy_score

trained_params = np.load("trained_qcnn_params.npy")

# Make predictions
predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, embedding_type) for x in X_test)]
predictions = np.argmax(predictions, axis=1)

# Compute accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"QCNN Classification Accuracy: {accuracy * 100:.2f}%")
