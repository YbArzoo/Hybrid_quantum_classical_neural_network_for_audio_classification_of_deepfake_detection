import os
import data
import Benchmarking
import CNN
import Hybrid_embedding_accuracy
import Hybrid_embedding_entanglement
import numpy as np
import QCNN_circuit
from qcnn_training import circuit_training

# Ensure spectrogram images exist
spectrogram_path = r"D:\BRACU\THESIS\QCNN CODE\QCNN-main\QCNN\spectrograms_images"
if not os.path.exists(spectrogram_path):
    print("Spectrogram images not found. Please generate them first!")
    exit()



# Step 1: Load and preprocess dataset
print("Loading and processing spectrogram dataset...")
# Pass the spectrogram path to load the dataset
X_train, X_test, Y_train, Y_test = data.data_load_and_process(spectrogram_path)


# Step 2: Train Classical CNN
print("Training CNN model...")
CNN.train_CNN(X_train, X_test, Y_train, Y_test)

# Step 3: Optimize Quantum Embeddings
print("Evaluating Quantum Embeddings...")
# Here, we call Hybrid_embedding_accuracy.Benchmarking_Hybrid_Accuracy()



# This step optimizes the embeddings and evaluates them based on your dataset
Unitary = 'U_SU4'
U_params = 15 
U_num_param = 15
circuit = 'QCNN'
Encodings = ['autoencoder16-4'] # Or use any other encodings that you want

# Debugging U_params before passing to Benchmarking_Hybrid_Accuracy
print(f"Debug: U_params = {U_params}, Expected params = {3 * U_params + 6}")  # Add debug here


Hybrid_embedding_accuracy.Benchmarking_Hybrid_Accuracy(spectrogram_path, Y_train, Unitary, U_num_param, Encodings, circuit, binary=True)

# Step 4: Evaluate Quantum Embeddings (this can stay as it is, or you can modify it based on the previous step)
# Hybrid_embedding_accuracy.evaluate_embedding(X_train, Y_train, embedding_type='Hybrid')

# Step 5: Analyze Quantum Entanglement
# print("🔹 Analyzing Quantum Entanglement...")
# Hybrid_embedding_entanglement.analyze_entanglement(X_train, embedding_type='Hybrid')

# Step 6: Train QCNN Model
print("Training QCNN...")
U = 'U_SU4'  # Select quantum unitary ansatz
U_params = 15
embedding_type = "Amplitude"

# Use direct function call from qcnn_training
loss_history, trained_params = circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit='QCNN', cost_fn='cross_entropy')

# Save trained QCNN model
np.save("trained_qcnn_params.npy", trained_params)
print("QCNN Training Completed!")

# Step 7: Evaluate QCNN
print("Evaluating QCNN on Test Data...")
trained_params = np.load("trained_qcnn_params.npy")

# ✅ Pass X_test, Y_test to Benchmarking
Benchmarking.run_benchmark(X_test, Y_test, trained_params, U, U_params, embedding_type)
