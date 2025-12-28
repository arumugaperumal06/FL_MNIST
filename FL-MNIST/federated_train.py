import torch
import torch.nn as nn
from data_loader import get_client_data
from clients.client1 import Client
from server.server import Server

# -----------------------------
# Step 1: Load data and split for clients
# -----------------------------
client_data_list, test_dataset = get_client_data()

# Initialize clients
clients = []
for i, data in enumerate(client_data_list):
    clients.append(Client(client_id=i+1, train_data=data))

# -----------------------------
# Step 2: Initialize server
# -----------------------------
server = Server()

# -----------------------------
# Step 3: Federated Training Loop
# -----------------------------
num_rounds = 5     # Number of federated rounds
local_epochs = 1   # Epochs each client trains locally

for r in range(num_rounds):
    print(f"\n--- Federated Round {r+1} ---")

    # 1️⃣ Send global weights to clients
    global_weights = server.get_global_weights()
    for client in clients:
        client.set_weights(global_weights)

    # 2️⃣ Each client trains locally
    for client in clients:
        client.train(epochs=local_epochs)

    # 3️⃣ Collect updated weights from clients
    client_weights = [client.get_weights() for client in clients]

    # 4️⃣ Server aggregates client weights
    server.aggregate(client_weights)

# -----------------------------
# Step 4: Evaluate Global Model
# -----------------------------
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


