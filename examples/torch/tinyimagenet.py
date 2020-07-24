import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import torchvision as tv
import numpy as np
import pickle 

batch_size = 256

class TinyVGG16(nn.Module):
    def __init__(self, num_classes = 200):
        super(TinyVGG16, self).__init__()
        self.features = nn.Sequential(
            # conv 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output

def load_data(filepath):
  with open(filepath, 'rb') as f:
    cat_dog_data = pickle.load(f)
    x_train = cat_dog_data['train']['data']
    y_train = cat_dog_data['train']['label']
    x_test = cat_dog_data['test']['data']
    y_test = cat_dog_data['test']['label']
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)

def get_accuracy(y_predict, y_true):
    return np.mean(np.equal(np.argmax(y_predict, axis=-1),
                            np.argmax(y_true, axis=-1)))

def prepare_data():
    x_train, y_train, x_test, y_test = load_data("/content/drive/My Drive/dataset/tinyimagenet.pkl")
    print(x_train.shape)
    x_train, y_train, x_test, y_test = torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size)
    return train_loader, test_loader

def main():
    device = 'cuda'
    model = TinyVGG16(200)
    model = model.to(device)
    train_loader, test_loader = prepare_data()
    epochs = 200
    log_interval = 500
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Train Phase
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            train_correct += torch.sum(preds == target)
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch_idx, loss))
        print('Accuracy on training set: {}'.format(100. * train_correct/len(train_loader.dataset)))

        # Test Phase
        correct = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.long().to(device)
                results = model(data)
                _, pred = torch.max(results, 1)
                correct += (pred == target).sum().item()
        print("On Test Set: Accuracy: {} % with {} corrects".format(100. * correct/len(test_loader.dataset), correct))
    torch.save(model.state_dict(), "/content/drive/My Drive/model/tinyimagenet.pth")
main()