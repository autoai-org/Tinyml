import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.utils.data import TensorDataset, DataLoader

import pickle

# Utilities
def load_data(filepath):
  with open(filepath, 'rb') as f:
    cat_dog_data = pickle.load(f)
    data = cat_dog_data['image']
    label = cat_dog_data['labels']
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.10, random_state=42)
    return x_train, y_train, x_test, y_test

def get_accuracy(y_predict, y_true):
    return np.mean(np.equal(np.argmax(y_predict, axis=-1),
                            np.argmax(y_true, axis=-1)))

x_train, y_train, x_test, y_test = load_data('dataset/cat_and_dog.pkl')

def preprocess_y(y_train, y_test):
  enc = OneHotEncoder(sparse=False, categories='auto')
  y_train = enc.fit_transform(y_train.reshape(len(y_train), -1))
  y_test = enc.transform(y_test.reshape(len(y_test), -1))
  return y_train, y_test

y_train, y_test = preprocess_y(y_train, y_test)

device = 'cuda'
model = vgg16(pretrained=False)
model.to(device)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

optimizer = torch.optim.SGD(lr=0.01)
criterion = nn.NLLLoss()

training_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test, y_test)

training_dataset = DataLoader(training_dataset, batch_size=4)
test_dataset = DataLoader(test_dataset, batch_size=4)

for epoch in range(5):
    for data, target in training_dataset:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("loss: {}".format(loss))

y_predict = model(x_test)
acc = get_accuracy(y_predict, y_test)
print('Testing Accuracy: {}%'.format(acc*100))