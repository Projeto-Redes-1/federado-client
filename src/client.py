
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import sys
import os
import pickle

class FederatedNet(torch.nn.Module):
    def __init__(self):
        # Define the CNN architecture with convolutional, pooling, and fully connected layers 
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 7)
        self.conv2 = torch.nn.Conv2d(20, 40, 7)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(2560, 10)
        self.non_linearity = torch.nn.functional.relu
        # Track layers for easy parameter extraction and update
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'linear': self.linear}

    def forward(self, x):
        x = self.non_linearity(self.conv1(x))
        x = self.non_linearity(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def get_parameters(self):
        #Extract model parameters as a dictionary of weights and biases
        return {name: {'weight': layer.weight.data.clone(), 'bias': layer.bias.data.clone()} for name, layer in self.track_layers.items()}

    def apply_parameters(self, parameters):
        # Update model parameters using a dictionary of weights and biases
        with torch.no_grad():
            for name in parameters:
                self.track_layers[name].weight.data.copy_(parameters[name]['weight'])
                self.track_layers[name].bias.data.copy_(parameters[name]['bias'])


class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.dataset = self.load_data()

    def load_data(self):
        #Load and partition CIFAR-10 dataset for the client
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        examples_per_client = len(full_dataset) // 3
        start_idx = self.client_id * examples_per_client
        end_idx = start_idx + examples_per_client
        return full_dataset[start_idx:end_idx]

    def train(self, parameters):
        #Train the model on the client's local dataset
        net = FederatedNet()
        net.apply_parameters(parameters)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True)

        for epoch in range(3):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

        return net.get_parameters()


if __name__ == "__main__":
    # Expecting client_id and parameters path
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <parameters_path>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    parameters_path = sys.argv[2]

    # Load parameters
    if os.path.exists(parameters_path):
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
    else:
        parameters = None

    client = Client(client_id)
    updated_parameters = client.train(parameters)

    # Save updated parameters
    output_path = f"client_{client_id}_parameters.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(updated_parameters, f)

    print(f"Client {client_id} training completed. Updated parameters saved to {output_path}")
