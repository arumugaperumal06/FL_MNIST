import torch
from model import CNNModel

class Server:
    def __init__(self):
        self.global_model = CNNModel()

    def get_global_weights(self):
        return self.global_model.state_dict()

    def aggregate(self, client_weights):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client[key] for client in client_weights], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
