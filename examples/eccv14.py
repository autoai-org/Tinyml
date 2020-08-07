#coding:utf-8
import os

import numpy as np
from numpy import savetxt
from numpy.core.fromnumeric import argmax

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tinynet.layers import Conv2D, Deconv2D, MaxPool2D, MaxUnpool2D, Softmax
from tinynet.models.inv_vgg11 import forward, inverse_vgg, load_weight
from tinynet.models.vgg11 import vgg11
'''
Constants
'''
data_path = './examples/assets/dataset/'
'''
Load images
'''


def load_images():
    datasets = []
    data_path = './examples/assets/dataset/'
    for filepath in os.listdir(data_path):
        image = Image.open(os.path.join(data_path, filepath))
        image = image.resize((224, 224), Image.ANTIALIAS)
        # transpose makes HxWxC to CxHxW
        image = np.asarray(image.convert('RGB')).transpose(2, 0, 1) / 255.
        datasets.append(image)
    return datasets


def load_single_image(id):
    if id > 19:
        raise ValueError("Not a valid image")
    for index, filepath in enumerate(os.listdir(data_path)):
        if index == id:
            image = Image.open(os.path.join(data_path, filepath))
            image = image.resize((224, 224), Image.ANTIALIAS)
            image = np.asarray(image.convert('RGB')).transpose(2, 0, 1) / 255.
            return image


'''
Load model
'''


def load_model():
    model = vgg11()
    model.load('./examples/assets/model.tnn.npy')
    return model


def load_inverse_model(vgg11):
    inv_vgg = inverse_vgg()
    inv_vgg = load_weight(inv_vgg, vgg11)
    return inv_vgg


'''
Perform Classification
'''


def classify():
    datasets = load_images()
    model = load_model()
    s = Softmax('softmax')
    correct = 0
    for each in datasets:
        each = each.reshape(1, 3, 224, 224)
        results = s(model.forward(each))
        result = np.argmax(results)
        correct += result
    print("Acc: {}/{}={}".format(correct, 20, correct / 20))


'''
Visualization
'''


def feed_store(x, model):
    features_maps = [None] * len(model.layers)
    pool_indices = [None] * len(model.layers)
    for index, layer in enumerate(model.layers):
        if isinstance(layer, MaxPool2D):
            x, max_indices = layer.forward(x)
            features_maps[index] = x
            pool_indices[index] = max_indices
        else:
            x = layer.forward(x)
            features_maps[index] = x
    return x, features_maps, pool_indices


def vis_layer(layer, feature_maps, inv_vgg, pool_locs):
    '''
    Visualizing the layer deconv result
    '''
    num_feat = feature_maps[layer + 1].shape[1]
    new_feat_map = feature_maps[layer + 1].copy()
    act_list = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = np.max(choose_map)
        act_list.append(activation)
    act_list = np.array(act_list)
    mark = np.argmax(act_list)
    choose_map = new_feat_map[0, mark, :, :]
    max_activation = np.max(choose_map)
    if mark == 0:
        new_feat_map[:, 1, :, :] = 0
    else:
        new_feat_map[:, mark, :, :] = 0
        if mark != feature_maps[layer + 1].shape[1] - 1:
            new_feat_map[:, mark + 1, :, :] = 0
    choose_map = np.where(choose_map == max_activation, choose_map,
                          np.zeros(choose_map.shape))
    new_feat_map[0, mark, :, :] = choose_map

    deconv_output = forward(new_feat_map, inv_vgg, layer, pool_locs)
    # transpose it from (C, H, W) back to (H, W, C)
    new_img = deconv_output[0]
    new_img = new_img.transpose(1, 2, 0)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    return new_img, int(max_activation)


def visualize(img_id, model, inv_model, image_arr):
    origin = (image_arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    '''
    classification
    '''
    image_input = image_arr.reshape(1, 3, 224, 224)
    output, feature_maps, pool_indices = feed_store(image_input, model)
    print("Predicted as {}".format(np.argmax(output)))
    '''
    inverse visualization
    '''
    plt.figure(num=None, figsize=(16, 12), dpi=120)
    plt.subplot(2, 4, 1)
    plt.title("Original Image, Label: {}".format(np.argmax(output)))
    plt.imshow(origin)
    for idx, layer in enumerate([7, 10, 12, 14, 17, 19, 21]):
        plt.subplot(2, 4, idx + 2)
        deconv_output, activation = vis_layer(layer, feature_maps, inv_model,
                                              pool_indices)
        plt.title("{}-th Layer, the max activation is {}".format(
            layer, activation))
        plt.imshow(deconv_output)
    plt.savefig("results/" + str(img_id) + ".jpg")


if __name__ == '__main__':
    model = load_model()
    inv_model = load_inverse_model(model)
    for i in range(20):
        img = load_single_image(i)
        visualize(i, model, inv_model, img)
