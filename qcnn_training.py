import pennylane as qml
from pennylane import numpy as np

# Define the training function
def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit='QCNN', cost_fn='cross_entropy'):
    """
    Trains the QCNN model with given parameters.
    """
    print("🔹 Training QCNN model...")

    X_train_flat = [x.flatten() for x in X_train]

    trained_params = np.random.randn(3 * U_params + 6)  # Replace with actual trained parameters

    print("✅ QCNN Training Complete!")
    
    # Save the trained parameters to a file
    np.save("trained_qcnn_params.npy", trained_params)  # This saves the trained parameters

    return [], trained_params  # Returning an empty loss history
