import os
from src import pca_models, reconstruction, data_loader
import numpy as np

COMPONENTS = 440
WIDTH = 92
HEIGHT = 112
TARGET_SIZE = (WIDTH, HEIGHT)
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
SHOW_TIME = False
PERSONAL_DATASET = False
SHOW_DATASET_SHAPE = False
SHOW_NUMBER_EFFICIENCY = False
SHOW_SPACE_EFFICIENCY = False
SHOW_MSE = False
RECONSTRUCTED_INDEXES = []

dataset_dir = os.path.join(ROOT_FOLDER, "olivetti_dataset")
dataset = data_loader.load_dataset(dataset_dir, target_size=TARGET_SIZE, )
# olivetti_dataset is (400 x 10,304): (40 people * 10 photos) x (92 * 112 pixels)
if PERSONAL_DATASET:
    personal_dataset_dir = os.path.join(ROOT_FOLDER, "personal_dataset")
    personal_dataset = data_loader.load_dataset(personal_dataset_dir, target_size=TARGET_SIZE, )
    dataset = np.concatenate((dataset, personal_dataset))
if SHOW_TIME:
    pca_models.pca_covariance(dataset, COMPONENTS, show_time=SHOW_TIME)  # Extremely slow without SVD and without dual PCA
    pca_models.pca_dual(dataset, COMPONENTS, show_time=SHOW_TIME)  # Slow without SVD
weights, W, mean = pca_models.pca_svd(dataset, COMPONENTS, show_time=SHOW_TIME)  # Fast with SVD
if SHOW_DATASET_SHAPE:
    print(f"Forma pesi: {weights.shape}")
    print(f"Forma W: {W.shape}")
    print(f"Forma media: {mean.shape}")
if SHOW_NUMBER_EFFICIENCY:
    original_values = dataset.size
    pca_values = weights.size + W.size + mean.size
    saving = (original_values - pca_values) / original_values * 100
    print(f"Numeri orginali: {original_values}, Numeri dopo PCA: {pca_values}")
    print(f"Risparmio di numeri: {saving:.2f}%")
if SHOW_SPACE_EFFICIENCY:
    original_values = dataset.nbytes
    pca_values = weights.nbytes + W.nbytes + mean.nbytes
    saving = (original_values - pca_values) / original_values * 100
    print(f"Spazio orginale: {original_values}, Spazio dopo PCA: {pca_values}")
    print(f"Risparmio di spazio: {saving:.2f}%")
reconstruction.eigen_faces(W, target_size=TARGET_SIZE, output_folder="output")
reconstructed_dataset = np.dot(weights, W.T) + mean
if SHOW_MSE:
    mse = np.mean((dataset - reconstructed_dataset) ** 2)
    original_variance = np.var(dataset)
    explained_variance_ratio = (1 - mse / original_variance) * 100 if original_variance > 0 else 0
    print(f"Informazione conservata: {explained_variance_ratio:.2f}%")
    print(f"Errore medio per pixel (MSE): {mse:.4f}")
for img_index in RECONSTRUCTED_INDEXES:
    reconstruction.comparison(dataset[img_index], reconstructed_dataset[img_index], img_index, target_size=TARGET_SIZE, output_folder="output")
test_dataset_dir = os.path.join(ROOT_FOLDER, "test_dataset")
data_test = data_loader.load_dataset(test_dataset_dir, target_size=TARGET_SIZE)
for i, test in enumerate(data_test):
    weights_test = np.dot(test - mean, W)
    reconstructed_data_test = np.dot(weights_test, W.T) + mean
    reconstruction.comparison(test, reconstructed_data_test, f"test {i + 1}", target_size=TARGET_SIZE, output_folder="output")
