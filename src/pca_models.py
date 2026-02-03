import numpy as np
from src.decorators import timer


@timer
def pca_covariance(dataset, components):
    """Compute PCA weights and mean from dataset using covariance"""
    mean = np.mean(dataset, axis=0)
    dataset_centred = (dataset - mean)
    covariance = np.cov(dataset_centred.T)  # Pixel values are columns
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(covariance)  # Covariance is symmetric
    eigenvectors = raw_eigenvectors[:, ::-1]  # Eigenvectors in descending order
    W = eigenvectors[:, :components]
    weights = np.dot(dataset_centred, W)
    return weights, W, mean


@timer
def pca_dual(dataset, components):
    """Compute PCA weights and mean from dataset using the dual method of PCA"""
    mean = np.mean(dataset, axis=0)
    dataset_centred = (dataset - mean)
    L = np.dot(dataset_centred, dataset_centred.T)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(L)
    eigenvectors = raw_eigenvectors[:, ::-1]
    W_raw = np.dot(dataset_centred.T, eigenvectors)
    W = W_raw / (np.linalg.norm(W_raw, axis=0) + 1e-15)
    W = W[:, :components]
    weights = np.dot(dataset_centred, W)
    return weights, W, mean


@timer
def pca_svd(dataset, components):
    """Compute PCA weights and mean from dataset using SVD"""
    mean = np.mean(dataset, axis=0)
    dataset_centred = (dataset - mean)
    _, _, VT = np.linalg.svd(dataset_centred, full_matrices=False)
    W = VT[:components, :].T
    weights = np.dot(dataset_centred, W)
    return weights, W, mean
