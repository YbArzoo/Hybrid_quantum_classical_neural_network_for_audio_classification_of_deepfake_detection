import QCNN_circuit
import numpy as np
import cv2

# Modify the function to accept `X_test`
def run_benchmark(X_test, Y_test, trained_params, U, U_params, embedding_type):
    print("Running Benchmarking on QCNN...")

    # Debug: Check the shape of X_test before processing
    print(f"Debug: X_test shape before QCNN = {X_test.shape}")

    # Ensure the input is reduced to 256 features
    X_test_flat = [cv2.resize(x, (16, 16)).flatten() for x in X_test]  # Resize to (16x16) = 256 features


    # Perform predictions using QCNN
    predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, embedding_type) for x in X_test_flat]
    
    # Convert probabilities to class labels
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == Y_test)
    print(f"QCNN Accuracy: {accuracy * 100:.2f}%")
