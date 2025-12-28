import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_client_data(num_clients=3, batch_size=32):
    """
    Loads MNIST dataset, splits it into `num_clients` parts.
    Returns:
        client_data_list: list of DataLoader for each client
        test_loader: DataLoader for test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST training and test sets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training data for clients
    train_size = len(train_dataset) // num_clients
    client_data_list = []
    for i in range(num_clients):
        start = i * train_size
        end = (i + 1) * train_size if i != num_clients - 1 else len(train_dataset)
        subset = Subset(train_dataset, range(start, end))
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_data_list.append(loader)

    # Test loader (for evaluating global model)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_data_list, test_loader
