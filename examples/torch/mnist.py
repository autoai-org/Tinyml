import torch
from torchvision import datasets, transforms

transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
])

test_set = datasets.MNIST('./dataset', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_set,batch_size=64)

for data, target in test_loader:
    print(data.shape)
    print(target.shape)