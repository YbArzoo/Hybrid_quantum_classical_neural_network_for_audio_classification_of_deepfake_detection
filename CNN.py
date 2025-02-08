import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
# from data import data_load_and_process  # Correct import
from data import data_load_and_process, IMG_SIZE  # Import IMG_SIZE from data.py


# Define batch size (Add this line)
batch_size = 25

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Fully connected layer (example)
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification (example)

    def forward(self, x):
        # Apply layers step by step during the forward pass
        x = self.pool(self.conv1(x))  # Apply conv1 and pooling
        x = torch.flatten(x, 1)  # Flatten the output to feed it to fully connected layers
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to fc1
        x = self.fc2(x)  # Output layer (no activation for binary classification)
        return x

# Now define the training and evaluation functions
def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def accuracy_test(predictions, labels):
    acc = 0
    for (p, l) in zip(predictions, labels):
        if p[0] >= p[1]:
            pred = 0
        else:
            pred = 1
        if pred == l:
            acc += 1
    return acc / len(labels)

def train_CNN(X_train, X_test, Y_train, Y_test, num_epochs=50):
    # Initialize your CNN model
    model = CNN()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ✅ Convert NumPy arrays to Torch tensors (DO NOT FLATTEN)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # ✅ Debug: Check X_train shape before reshaping
    print(f"🔹 Debug: X_train shape before reshaping: {X_train.shape}")  # Check if (N,64,64)

    # Ensure X_train and X_test have the right shape before reshaping
    if len(X_train.shape) == 3:  # Should be (num_samples, 64, 64)
        X_train = X_train.reshape(-1, 1, 64, 64)  # Convert to (batch, channels, height, width)
        X_test = X_test.reshape(-1, 1, 64, 64)
    else:
        print(f"⚠️ Warning: Unexpected shape {X_train.shape}, skipping reshaping.")

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, torch.tensor(Y_train, dtype=torch.long))  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        if epoch % 10 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == torch.tensor(Y_test, dtype=torch.long)).float().mean()
        print(f'✅ Accuracy on test set: {accuracy.item() * 100:.2f}%')




def Benchmarking_CNN(dataset, classes, Encodings, Encodings_size, binary, optimizer):
    for i in range(len(Encodings)):
        Encoding = Encodings[i]
        input_size = Encodings_size[i]
        final_layer_size = int(input_size / 4)

        print(f"Using dataset path: {dataset}")  # Debug print to check dataset path

        # Pass the correct dataset path
        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset)  # Correct dataset path

        model = CNN()  # Initialize the CNN model
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
        elif optimizer == 'nesterov':
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

        loss_history = []
        for it in range(steps):
            batch_idx = np.random.randint(0, len(X_train), batch_size)
            X_train_batch = np.array([X_train[i] for i in batch_idx])
            Y_train_batch = np.array([Y_train[i] for i in batch_idx])


            print(f"Debug: X_train shape before reshaping: {X_train.shape}")
            # Reshape input data to be in the form (batch_size, 1, height, width)
            X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.float32)
            X_train_batch_torch = X_train_batch_torch.view(batch_size, 1, IMG_SIZE[0], IMG_SIZE[1])  # Correct reshaping

            Y_train_batch_torch = torch.tensor(Y_train_batch, dtype=torch.long)
            
            # For the test set, ensure it is also reshaped correctly
            X_test_torch = torch.tensor(X_test, dtype=torch.float32)
            X_test_torch = X_test_torch.view(len(X_test), 1, IMG_SIZE[0], IMG_SIZE[1])  # Correct reshaping for test set

            Y_pred_batch_torch = model(X_train_batch_torch)

            loss = criterion(Y_pred_batch_torch, Y_train_batch_torch)
            loss_history.append(loss.item())
            if it % 10 == 0:
                print("[iteration]: %i, [LOSS]: %.6f" % (it, loss.item()))

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Ensure the test data is reshaped correctly too
            X_test_torch = torch.tensor(X_test, dtype=torch.float32)
            X_test_torch = X_test_torch.view(len(X_test), 1, IMG_SIZE[0], IMG_SIZE[1])  # Correct reshaping for test set

            Y_pred = model(X_test_torch).detach().numpy()
            accuracy = accuracy_test(Y_pred, Y_test)
            N_params = get_n_params(model)

        # Write results to file
        with open('Result/result_CNN.txt', 'a') as f:
            f.write(f"Loss History for CNN with {Encoding}:\n")
            f.write(f"{loss_history}\n")
            f.write(f"Accuracy for CNN with {Encoding} {optimizer}: {accuracy}\n")
            f.write(f"Number of Parameters used to train CNN: {N_params}\n\n")



steps = 200
dataset = r'D:\BRACU\THESIS\QCNN CODE\QCNN-main\QCNN\spectrograms_images'  # Correct dataset path
classes = [0, 1]
binary = False
Encodings = ['pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact']
Encodings_size = [8, 8, 16, 16]

for i in range(5):
    Benchmarking_CNN(dataset=dataset, classes=classes, Encodings=Encodings, Encodings_size=Encodings_size,
                     binary=binary, optimizer='adam')
