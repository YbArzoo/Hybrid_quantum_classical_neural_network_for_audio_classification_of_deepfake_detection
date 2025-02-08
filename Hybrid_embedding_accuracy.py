# This module implements the measurement method of the entangling capability
import pennylane as qml
import QCNN_circuit
import unitary
import embedding
import numpy as np
import data
import Training
import cv2
from CNN import accuracy_test



dev = qml.device('default.qubit', wires=8)

@qml.qnode(dev)
def QCNN_partial_trace(X, params, embedding_type='Angular-Hybrid4', qubit_index=0):
    embedding.data_embedding(X, embedding_type)
    QCNN_circuit.conv_layer1(unitary.U_SU4, params)
    return qml.density_matrix(wires=qubit_index)

def Meyer_Wallach(X, params, embedding_type):
   n = 8
   measure = 0
   for j in range(n):
        rho = QCNN_partial_trace(X, params, embedding_type, qubit_index=j)
        rho_squared = np.matmul(rho, rho)
        rho_squared_traced = np.matrix.trace(rho_squared)
        measure = measure + 1/2 * (1 - rho_squared_traced)
   return measure * 4 / n

# A function to convert encoding method to embedding
def Encoding_to_Embedding(encoding_method):
    if encoding_method == 'Amplitude':
        return 'Amplitude embedding'
    elif encoding_method == 'autoencoder16-4':
        return 'Autoencoder 16-4 embedding'
    # Add more encoding methods as needed
    elif encoding_method == 'autoencoder30-3':
        return 'Autoencoder 30-3 embedding'
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")





def Benchmarking_Hybrid_Accuracy(spectrogram_path, classes, Unitary, U_num_param, Encodings, circuit, binary=True):
    U = Unitary
    U_params = U_num_param
    J = len(Encodings)
    best_trained_params_list = []

    for j in range(J):
        Encoding = Encodings[j]
        Embedding = Encoding_to_Embedding(Encoding)
        f = open('Result/Hybrid_result_' + str(Encoding) + '.txt', 'a')
        trained_params_list = []
        accuracy_list = []
        
        for n in range(5):
            X_train, X_test, Y_train, Y_test = data.data_load_and_process(spectrogram_path)

            
            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding)

            # Train the QCNN
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            # 🔹 Debugging trained parameter size
            print(f"Debug: Trained parameters size = {len(trained_params)}")  # ✅ Add debug here
            
            # Predict using trained QCNN
            # Resize X_test to (16x16) before passing it to QCNN
            X_test_flat = [cv2.resize(x, (16, 16)).flatten() for x in X_test]  # Resize to 256 features
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test_flat]

            accuracy = accuracy_test(predictions, Y_test)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            trained_params_list.append(trained_params)
            accuracy_list.append(accuracy)

            f.write("Trained Parameters: \n")
            f.write(str(trained_params))
            f.write("\n")
            f.write("Accuracy: \n")
            f.write(str(accuracy))
            f.write("\n")

        index = accuracy_list.index(max(accuracy_list))
        best_trained_params_list.append(trained_params_list[index])

    f.close()
    return best_trained_params_list

def Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, N_samples, best_trained_params_list):
    for i in range(len(Encodings)):
        Encoding = Encodings[i]
        print("Processing " + str(Encoding) + ".....\n")
        Embedding = Encoding_to_Embedding(Encoding)
        best_trained_params = best_trained_params_list[i]
        best_trained_params = best_trained_params[:15]

        # X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset=dataset, classes=classes, feature_reduction=Encoding, binary=True)
        
        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset)

        random_index = np.random.randint(0, len(X_test), N_samples)
        X_test = X_test[random_index]

        entanglement_measure = [Meyer_Wallach(X, best_trained_params, Embedding) for X in X_test]
        mean_entanglement_measure = np.mean(entanglement_measure)
        stdev_entanglement_measure = np.std(entanglement_measure)

        f = open('Result/Hybrid_result_' + str(Encoding) + '.txt', 'a')
        f.write("\n")
        f.write("Entanglement measure Mean: ")
        f.write(str(mean_entanglement_measure))
        f.write("\n")
        f.write("Entanglement measure Standard Deviation: ")
        f.write(str(stdev_entanglement_measure))
        f.write("\n")
        f.close()

dataset = r"D:\BRACU\THESIS\QCNN CODE\QCNN-main\QCNN\spectrograms_images"

classes = [0,1]
Unitary = 'U_SU4'
U_num_param = 15
circuit = 'QCNN'
N_samples = 1000




Encodings = ['autoencoder30-3']
