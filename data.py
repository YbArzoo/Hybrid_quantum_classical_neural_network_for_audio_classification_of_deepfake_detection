import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)  # Keep all images 64x64

def data_load_and_process(spectrogram_path):
    """
    Loads spectrogram images from the given path and prepares them for training.
    Returns: X_train, X_test, Y_train, Y_test
    """

    fake_path = os.path.join(spectrogram_path, "FAKE")  # FAKE folder
    real_path = os.path.join(spectrogram_path, "REAL")  # REAL folder

    data = []

    # Load FAKE images
    for file in os.listdir(fake_path):
        if file.endswith('.png'):
            img_path = os.path.join(fake_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)  # ✅ Resize all images to (64,64)
            img = img / 255.0  # Normalize pixel values
            data.append((img, 0))  # Label 0 for FAKE

    # Load REAL images
    for file in os.listdir(real_path):
        if file.endswith('.png'):
            img_path = os.path.join(real_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)  # ✅ Resize all images to (64,64)
            img = img / 255.0  # Normalize pixel values
            data.append((img, 1))  # Label 1 for REAL

    # Shuffle data
    np.random.shuffle(data)

    # ✅ Debugging step: Ensure all images are 64x64 before conversion
    print(f"🔹 Debug: Loaded {len(data)} samples before converting to NumPy array.")
    for i, (img, label) in enumerate(data[:5]):  # Print first 5 for debugging
        print(f"🔹 Debug: Image {i} shape = {img.shape}, Label = {label}")

    # ✅ Convert list of images into a NumPy array with a fixed shape
    X = np.array([item[0] for item in data], dtype=np.float32)  # Ensure float32 for CNN
    Y = np.array([item[1] for item in data], dtype=np.int64)  # Ensure labels are integers

    print(f"✅ Debug: Final dataset shape X: {X.shape}, Y: {Y.shape}")  # Should be (num_samples, 64, 64)

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test
