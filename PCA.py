import numpy as np
from Decoratori import time_count


@time_count
def pca_covariance(Dataset, n_components):
    mean = np.mean(Dataset, axis=0)
    Dataset_Centred = (Dataset - mean)
    Covariance = np.cov(Dataset_Centred.T)  # Variabili su colonne (pixel)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(Covariance)
    eigenvectors = raw_eigenvectors[:, ::-1]
    W = eigenvectors[:, :n_components]
    Dataset_PCA = np.dot(Dataset_Centred, W)
    return Dataset_PCA, W, mean


@time_count
def pca_svd(Dataset, n_components):
    mean = np.mean(Dataset, axis=0)
    Dataset_Centred = (Dataset - mean)
    _, _, VT = np.linalg.svd(Dataset_Centred, full_matrices=False)
    W = VT[:n_components, :].T
    Dataset_PCA = np.dot(Dataset_Centred, W)
    return Dataset_PCA, W, mean
