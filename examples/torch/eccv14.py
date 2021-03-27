import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def forward_img(model, x):
    return model(x)


def build_reversed_model(x, model):
    layer_lists = []
    pooling_indices = []
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            x = layer(x)
            pooling_indices.append([0])
            deconv_layer = nn.ConvTranspose2d(layer.out_channels,
                                              layer.in_channels,
                                              layer.kernel_size, layer.stride,
                                              layer.padding)
            deconv_layer.weight = layer.weight
            layer_lists.append(deconv_layer)
        if isinstance(layer, nn.ReLU):
            x = layer(x)
            pooling_indices.append([0])
            layer_lists.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            x, index = layer(x)
            pooling_indices.append(index)
            unpool_layer = nn.MaxUnpool2d(kernel_size=layer.kernel_size,
                                          stride=layer.stride,
                                          padding=layer.padding)
            unpool_layer.return_indices = False
            layer_lists.append(unpool_layer)
    layer_lists.reverse()
    return nn.Sequential(*layer_lists), pooling_indices


def backward_featuremaps(y, model, layer_count, pooling_indices,
                         num_of_layers):
    output = y
    print('starting backward from {}'.format(num_of_layers - layer_count - 1))
    for i, layer in enumerate(model):
        if (i >= (num_of_layers - layer_count - 1)):
            print('backward layer {}: {}'.format(i, layer))
            if isinstance(layer, nn.MaxUnpool2d):
                print(pooling_indices[-1].shape)
                print(output[0].shape)
                output = layer(
                    output[0],
                    pooling_indices[num_of_layers - layer_count - 1])
            else:
                output = layer(output)
    return output


def visualize(id_of_layer, img):
    npimg = img[0].data.numpy()
    npimg = ((npimg - npimg.min()) * 255 /
             (npimg.max() - npimg.min())).astype('uint8')
    npimg = np.transpose(npimg, (1, 2, 0))
    path = "examples/output/" + str(id_of_layer) + "th-layer_reconstructed.png"

    # plt.imshow(npimg)
    # plt.show()
    plt.imsave(path, npimg)


if __name__ == '__main__':
    img = Image.open('examples/images/deer.jpg')
    resized_img = img.resize((224, 224))
    resized_img = np.array(resized_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(resized_img).unsqueeze_(0)

    model = models.vgg16(pretrained=True).eval()
    num_of_layers = 0
    for i, layer in enumerate(model.features):
        num_of_layers = num_of_layers + 1
        if (isinstance(layer, nn.MaxPool2d)):
            layer.return_indices = True
    demodel, pooling_indices = build_reversed_model(input_img, model)
    print(model.features)
    print(demodel)
    print(num_of_layers)
    raw_features = input_img
    for i, layer in enumerate(model.features):
        print('forward layer {}: {}'.format(i, layer))
        raw_features = layer(raw_features)
        reconstructed_img = backward_featuremaps(raw_features, demodel, i,
                                                 pooling_indices,
                                                 num_of_layers)
        visualize(i, reconstructed_img)
