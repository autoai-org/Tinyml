import numpy as np
import pickle

def load_tiny_imagenet(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return np.asarray(data['train']['data']), np.asarray(data['train']['label']), np.asarray(data['test']['data']), np.asarray(data['test']['label']), 