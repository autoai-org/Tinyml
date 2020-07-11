import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def forward_img(model, x, layer_max_count):
    deconv_layers_list = []
    unpool_layers_list = []

    layer_count = 0

    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            B, C, H, W = x.shape
            x = layer(x)
            deconv_layer = nn.ConvTranspose2d(layer.out_channels, C, layer.kernel_size, layer.stride, layer.padding)
            deconv_layer.weight = layer.weight
            deconv_layers_list.append(deconv_layer)

        if isinstance(layer, torch.nn.ReLU):
            x = layer(x)
            deconv_layers_list.append(layer)

        if isinstance(layer, torch.nn.MaxPool2d):
            x, index = layer(x)
            unpool_layers_list.append(index)
            unpool_layer = torch.nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                                padding=layer.padding)
            deconv_layers_list.append(unpool_layer)

        layer_count += 1
        if layer_max_count == layer_count:
            break

    return x, deconv_layers_list, unpool_layers_list


def filter_feature_maps(raw_feature_maps):
    feature_maps = raw_feature_maps[0]
    feature_maps_total_num = feature_maps.shape[0]

    activation_list = []
    for i in range(feature_maps_total_num):
        activation_val = torch.max(feature_maps[i, :, :])
        activation_list.append(activation_val.item())

    max_map_num = np.argmax(np.array(activation_list))
    max_map = feature_maps[max_map_num, :, :]
    max_activation_val = torch.max(max_map)
    max_map = torch.where(max_map == max_activation_val,
                          max_map,
                          torch.zeros(max_map.shape)
                          )

    for i in range(feature_maps_total_num):
        if i != max_map_num:
            feature_maps[i, :, :] = 0
        else:
            feature_maps[i, :, :] = max_map

    return feature_maps.unsqueeze_(0)


def backward_feature_maps(y, deconv_layers_list, unpool_layers_list):
    for layer in reversed(deconv_layers_list):
        if isinstance(layer, nn.MaxUnpool2d):
            y = layer(y, unpool_layers_list.pop())
        else:
            y = layer(y)

    return y


def visualize(layer_max_count, img):
    npimg = img[0].data.numpy()
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype('uint8')
    npimg = np.transpose(npimg, (1, 2, 0))
    path = "examples/output/" + str(layer_max_count) + "th-layer.png"

    plt.imshow(npimg)
    plt.show()
    plt.imsave(path, npimg)


if __name__ == '__main__':

    img = Image.open('examples/images/deer.jpg')
    # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    resized_img = img.resize((224,224))
    resized_img = np.array(resized_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(resized_img).unsqueeze_(0)

    model = models.vgg16(pretrained=True).eval()
    visualize_layer_indices = []

    for i, layer in enumerate(model.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            layer.return_indices = True
            visualize_layer_indices.append(i)

    for layer_max_count in visualize_layer_indices:
        print("layer...%s" % layer_max_count)
        raw_feature_maps, deconv_layers_list, unpool_layers_list = forward_img(model, input_img, layer_max_count)
        #filtered_feature_maps = filter_feature_maps(raw_feature_maps)
        #reproducted_img = backward_feature_maps(filtered_feature_maps, deconv_layers_list, unpool_layers_list)
        reproducted_img = backward_feature_maps(raw_feature_maps, deconv_layers_list, unpool_layers_list)
        visualize(layer_max_count, reproducted_img)
