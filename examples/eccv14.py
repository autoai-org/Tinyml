#coding:utf-8
from tinynet.models.vgg11 import vgg11
from PIL import Image
import os
import numpy as np

'''
Load images
'''
def load_images():
    datasets = []
    data_path = './examples/assets/dataset/'
    for filepath in os.listdir(data_path):
        image = np.asarray(Image.open(os.path.join(data_path,filepath)).convert('RGB')).transpose(2, 0, 1)
        datasets.append(image)
    return datasets

'''
Load model
'''
def load_model():
    model = vgg11()
    model.load('./examples/assets/model-36.tnn')
    return model

'''
Perform Classification
'''
def classify():
    datasets = load_images()
    model = load_model()
    for each in datasets:
        results = model.predict(each)
        print(results)

if __name__=='__main__':
    classify()