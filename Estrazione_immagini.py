import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_dataset(root_folder):
    ''' Estra le immagini dalla cartella root_folder e le salva in faces come valori numerici '''
    faces = []  # Matrice numero foto * dimensione foto
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pgm"):
                path = os.path.join(subdir, file)
                img = plt.imread(path)
                img_vector = img.flatten()  # La PCA usa vettori non matrici
                faces.append(img_vector)
    X = np.array(faces)
    return X


def load_foto_test(root_folder, w, h):
    foto_test_path = os.path.join(root_folder, "Foto_test.png")
    img = Image.open(foto_test_path).convert('L')
    img_resized = img.resize((w, h))
    data_test = np.array(img_resized).flatten()
    return data_test

