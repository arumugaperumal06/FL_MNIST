import torch
from torch import nn, optim
from model import CNNModel

class Client:
    def __init__(self, client_id, train_data):
        self.id = client_id
        self.train_data = train_data
        self.model = CNNModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.train_data:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_weights):
        self.model.load_state_dict(new_weights)
