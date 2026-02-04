import os
from cmath import inf
from src import pca_models, reconstruction, data_loader, face_recognition
import numpy as np

COMPONENTS = 5
WIDTH = 92
HEIGHT = 112
TARGET_SIZE = (WIDTH, HEIGHT)
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
SHOW_TIME = False
PERSONAL_DATASET = False
SHOW_DATASET_SHAPE = False
SHOW_VALUES_EFFICIENCY = False
SHOW_SPACE_EFFICIENCY = False
SHOW_MSE = False
EXECUTE_TEST_DATASET = False
EXECUTE_FACE_RECOGNITION = True
RECONSTRUCTED_INDEXES = []  # Indexes need to be between 0 and 359

dataset_dir = os.path.join(ROOT_FOLDER, "olivetti_dataset")
train_dataset = data_loader.load_dataset(dataset_dir, target_size=TARGET_SIZE)
# olivetti_dataset is (360 x 10,304): (40 people * 9 photos) x (92 * 112 pixels)
# one photo per person has been excluded and placed in recognition_dataset
if PERSONAL_DATASET:
    personal_dataset_dir = os.path.join(ROOT_FOLDER, "personal_dataset")
    personal_dataset = data_loader.load_dataset(personal_dataset_dir, target_size=TARGET_SIZE)
    if personal_dataset:
        train_dataset = np.concatenate((train_dataset, personal_dataset))
if SHOW_TIME:
    pca_models.pca_covariance(train_dataset, COMPONENTS, show_time=SHOW_TIME)  # Extremely slow without SVD and without dual PCA
    pca_models.pca_dual(train_dataset, COMPONENTS, show_time=SHOW_TIME)  # Slow without SVD
weights_train, W, mean = pca_models.pca_svd(train_dataset, COMPONENTS, show_time=SHOW_TIME)  # Fast with SVD
if SHOW_DATASET_SHAPE:
    print(f"Forma pesi: {weights_train.shape}")
    print(f"Forma W: {W.shape}")
    print(f"Forma media: {mean.shape}")
if SHOW_VALUES_EFFICIENCY:
    original_values = train_dataset.size
    pca_values = weights_train.size + W.size + mean.size
    saving = (original_values - pca_values) / original_values * 100
    print(f"Valori orginali: {original_values}, Valori dopo PCA: {pca_values}")
    print(f"Risparmio di valori: {saving:.2f}%")
if SHOW_SPACE_EFFICIENCY:
    original_values = train_dataset.nbytes
    pca_values = weights_train.nbytes + W.nbytes + mean.nbytes
    saving = (original_values - pca_values) / original_values * 100
    print(f"Spazio orginale: {original_values}, Spazio dopo PCA: {pca_values}")
    print(f"Risparmio di spazio: {saving:.2f}%")
reconstruction.eigen_faces(W, target_size=TARGET_SIZE, output_folder="output")
reconstructed_dataset = np.dot(weights_train, W.T) + mean
if SHOW_MSE:
    mse = np.mean((train_dataset - reconstructed_dataset) ** 2)
    original_variance = np.var(train_dataset)
    nsr = mse / original_variance if original_variance > 0 else 0
    explained_variance_ratio = (1 - nsr) * 100 if original_variance > 0 else 0
    print(f"Informazione conservata: {explained_variance_ratio:.2f}%")
    print(f"Errore medio per pixel (MSE): {mse:.4f}")
    print(f"SNR: {1 / nsr if nsr != 0 else float(inf):.4f}")
for img_index in RECONSTRUCTED_INDEXES:
    reconstruction.comparison(train_dataset[img_index], reconstructed_dataset[img_index], img_index, target_size=TARGET_SIZE, output_folder="output")
if EXECUTE_TEST_DATASET:
    test_dataset_dir = os.path.join(ROOT_FOLDER, "test_dataset")
    data_test = data_loader.load_dataset(test_dataset_dir, target_size=TARGET_SIZE)
    for i, test in enumerate(data_test):
        weights_test = np.dot(test - mean, W)
        reconstructed_data_test = np.dot(weights_test, W.T) + mean
        reconstruction.comparison(test, reconstructed_data_test, f"test {i + 1}", target_size=TARGET_SIZE, output_folder="output")
if EXECUTE_FACE_RECOGNITION:
    recognition_dataset_dir = os.path.join(ROOT_FOLDER, "recognition_dataset")
    recognition_dataset = data_loader.load_dataset(recognition_dataset_dir, target_size=TARGET_SIZE)
    for i, image in enumerate(recognition_dataset):
        weights_image = np.dot(image - mean, W)
        face_recognition.recognition(weights_train, weights_image, train_dataset, image, i+1, W, mean, target_size=TARGET_SIZE, output_folder="output")
