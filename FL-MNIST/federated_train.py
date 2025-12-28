import torch
import torch.nn as nn
from data_loader import get_client_data
from clients.client1 import Client
from server.server import Server


client_data_list, test_dataset = get_client_data()

# Initialize clients
clients = []
for i, data in enumerate(client_data_list):
    clients.append(Client(client_id=i+1, train_data=data))


server = Server()


num_rounds = 5     # Number of federated rounds
local_epochs = 1   # Epochs each client trains locally

for r in range(num_rounds):
    print(f"\n--- Federated Round {r+1} ---")

   
    global_weights = server.get_global_weights()
    for client in clients:
        client.set_weights(global_weights)

    
    for client in clients:
        client.train(epochs=local_epochs)

    
    client_weights = [client.get_weights() for client in clients]


    server.aggregate(client_weights)


def evaluate(model,test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"\nGlobal Model Accuracy: {100 * correct / total:.2f}%")

evaluate(server.global_model, test_dataset)



