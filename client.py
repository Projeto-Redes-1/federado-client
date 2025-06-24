import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import sys
import os
import pickle
import time  # Para medir tempo de execução

from torch.utils.data import Subset #para divisão


# Define a arquitetura da rede neural (mesma usada pelo servidor)
class FederatedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()  # Inicializa o módulo base do PyTorch

        # Camadas do modelo (mesma estrutura do servidor)
        self.conv1 = torch.nn.Conv2d(3, 20, 7)
        self.conv2 = torch.nn.Conv2d(20, 40, 7)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        #self.linear = torch.nn.Linear(2560, 10)
        self.linear = torch.nn.Linear(4000, 10) #mudei pra se adequar
        self.non_linearity = torch.nn.functional.relu

        # Camadas que queremos rastrear para obter/aplicar os parâmetros
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'linear': self.linear}

    def forward(self, x):
        # Passagem de dados pela rede
        x = self.non_linearity(self.conv1(x))
        x = self.non_linearity(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def get_parameters(self):
        # Extrai os parâmetros (pesos e bias) das camadas rastreadas
        return {
            name: {
                'weight': layer.weight.data.clone(),
                'bias': layer.bias.data.clone()
            } for name, layer in self.track_layers.items()
        }

    def apply_parameters(self, parameters):
        # Aplica novos parâmetros ao modelo (peso e bias)
        with torch.no_grad():
            for name in parameters:
                self.track_layers[name].weight.data.copy_(parameters[name]['weight'])
                self.track_layers[name].bias.data.copy_(parameters[name]['bias'])


# Classe que representa o cliente federado
class Client:
    def __init__(self, client_id):
        self.client_id = client_id  # ID do cliente (0, 1, 2...)
        self.dataset = self.load_data()  # Carrega o subconjunto dos dados

    def load_data(self):
        # Carrega e divide o dataset CIFAR-10 igualmente entre os clientes
        print(f"📦 Cliente {self.client_id}: carregando dados CIFAR-10...")
        transform = transforms.Compose([transforms.ToTensor()])  # Transformação básica
        full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Divide o dataset igualmente entre 3 clientes
        examples_per_client = len(full_dataset) // 3
        start_idx = self.client_id * examples_per_client
        end_idx = start_idx + examples_per_client
        #client_dataset = full_dataset[start_idx:end_idx] dando bronca pra pegar subset
        indices = list(range(start_idx, end_idx))
        client_dataset = Subset(full_dataset, indices)

        print(f"✅ Cliente {self.client_id}: dados carregados ({len(client_dataset)} exemplos).")
        return client_dataset

    def train(self, parameters):
        # Treina o modelo local com os parâmetros recebidos do servidor
        print(f"🧠 Cliente {self.client_id}: iniciando treinamento local...")

        start_time = time.time()  # Início da contagem de tempo

        net = FederatedNet()  # Inicia mas se n tiver, resole tbm
        if parameters:
            print(f"🔁 Cliente {self.client_id}: aplicando parâmetros recebidos do servidor.")
            net.apply_parameters(parameters)
        else:
            print(f"⚠️ Cliente {self.client_id}: parâmetros ausentes, iniciando com pesos aleatórios.")

        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # Otimizador SGD
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        # ⚡️ Início do treinamento
        print(f"⏳ Cliente {self.client_id}: iniciando treinamento por 3 épocas...\n")

        # Treinamento por 3 épocas
        for epoch in range(3):
            print(f"📚 Cliente {self.client_id}: Época {epoch + 1}/3")
            total_loss = 0
            batch_count = 0

            for inputs, labels in dataloader:
                batch_idx = batch_count  # para acompanhar o número do batch
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # ⚡️ Print progress a cada 10 batches
                if batch_count % 10 == 0:
                    print(f"   🔹 Batch {batch_count}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / batch_count
            print(f"✅ Cliente {self.client_id}: Época {epoch + 1} finalizada. Loss médio: {avg_loss:.4f}\n")

        print(f"🎯 Cliente {self.client_id}: treinamento finalizado com sucesso.")
        print(f"⏱️ Cliente {self.client_id}: tempo total de treinamento: {time.time() - start_time:.2f} segundos.")

        return net.get_parameters()  # Retorna os parâmetros atualizados


# Execução principal
if __name__ == "__main__":
    # Espera receber dois argumentos: id do cliente e caminho dos parâmetros globais
    if len(sys.argv) != 3:
        print("❌ Uso correto: python client.py <client_id> <parameters_path>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    parameters_path = sys.argv[2]

    # Tenta carregar os parâmetros recebidos do servidor
    print(f"📥 Cliente {client_id}: lendo parâmetros de {parameters_path}...")
    if os.path.exists(parameters_path):
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
        print(f"✅ Cliente {client_id}: parâmetros carregados.")
    else:
        print(f"⚠️ Cliente {client_id}: arquivo {parameters_path} não encontrado. Usando modelo aleatório.")
        parameters = None

    # Inicializa cliente e treina localmente
    client = Client(client_id)
    updated_parameters = client.train(parameters)

    # Salva os novos parâmetros treinados localmente
    output_path = f"client_{client_id}_parameters.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(updated_parameters, f)

    print(f"📤 Cliente {client_id}: parâmetros atualizados salvos em {output_path}")
