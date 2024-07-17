import os
import shutil
import pickle
import numpy as np


def normalize(img):
    img_copy  = img.copy()
    img_copy  = img_copy.astype('float64') 
    img_min   = np.min(img_copy)
    img_max   = np.max(img_copy)
    img_copy  = (img_copy - img_min)/(img_max - img_min)
    return img_copy

def create_circle_filter(N1, N2, R):
    n1   = np.linspace(-N1/2, N1/2, N1, dtype = 'float64')
    n2   = np.linspace(-N2/2, N2/2, N2, dtype = 'float64')
    n1, n2 = np.meshgrid(n1, n2)
    img  = (np.sqrt((n1/R)**2 + (n2/R)**2) <= 1)
    img  = img.astype('float64')
    return img

def convert_8bits(img):
    img_copy  = normalize(img)
    img_copy  = np.round(255*img_copy).astype('uint8')
    return img_copy

def save(filename, *args):
    glob = globals()
    d = {}
    for v in args:
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        pickle.dump(d, f)

def remove_files(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        print(f"Todos los archivos en {directory_path} han sido eliminados.")
    except Exception as e:
        print(f"Se ha encontrado un error: {e}")