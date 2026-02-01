import Estrazione_immagini
import os
import PCA
import Ricostruzione
from Estrazione_immagini import load_foto_test
import numpy as np

n_components = 400
width = 92
height = 112
project_root = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_root, "att_faces")
Dataset = Estrazione_immagini.load_dataset(dataset_dir)
# Dataset è una matrice (40*10)x(92*112) ovvero (40 persone * 10 foto)x(larghezza*altezza)
# Dataset_PCA, W, mean = PCA.pca_covariance(Dataset, 40) (Lento senza svd)
Dataset_PCA, W, mean = PCA.pca_svd(Dataset, n_components)  # Veloce con svd
print(f"Dataser shape: {Dataset_PCA.shape}")
print(f"W shape: {W.shape}")
print(f"mean shape: {mean.shape}")
Original_values = Dataset.size
PCA_values = Dataset_PCA.size + W.size + mean.size
print(f"Original values: {Original_values}, PCA values: {PCA_values}")
print(f"Risparmio: {(Original_values - PCA_values)/Original_values*100:.2f}%")
Ricostruzione.eigen_faces(W, 112, 92)
Dataset_reconstructed = Ricostruzione.ricostruttore(Dataset_PCA, W, mean)
img_index = 53
Ricostruzione.comparison(Dataset[img_index], Dataset_reconstructed[img_index], height, width, img_index)
Data_test = load_foto_test(project_root, width, height)
weights = np.dot(Data_test - mean, W)
Data_test_reconstructed = np.dot(weights, W.T) + mean
Ricostruzione.comparison(Data_test, Data_test_reconstructed, height, width, "test")
