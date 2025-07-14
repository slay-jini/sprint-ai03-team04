import torch
from torchvision import datasets, transforms

def get_mnist_data_loader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data/mnist', train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader 