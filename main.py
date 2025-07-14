import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import SimpleMLP
from utils.data_loader import get_mnist_data_loader

def train(epochs):
    # 데이터 로더
    train_loader = get_mnist_data_loader(train=True)
    test_loader = get_mnist_data_loader(train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device type :', device)

    # 모델, 손실 함수, 옵티마이저
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten the input
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

    # 테스트 루프
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test Accuracy: {100. * correct / len(test_loader.dataset)}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple MLP on MNIST')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    args = parser.parse_args()
    train(args.epochs) 