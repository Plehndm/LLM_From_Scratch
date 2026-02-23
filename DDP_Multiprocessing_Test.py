# Project: https://github.com/Plehndm/LLM_From_Scratch

# Imports to bring in previous neural network from notebook
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# New imports
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# Function to initialize a distributed process group (1 process / GPU)
# Allows communication among processes
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    # Here, we assume all GPUs are on the same machine
    os.environ["MASTER_ADDR"] = "127.0.0.1" # Address of the main node

    os.environ["MASTER_PORT"] = "12345" # Any free port on the machine

    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
       
        # Windows users may have to use "gloo" instead of "nccl" as backend
        init_process_group(
            backend="gloo",       # gloo: Facebook Collective Communication Library
            rank=rank,            # rank: refers to the index of the GPU we want to use
            world_size=world_size # world_size: number of GPUs to use
        )
    else:
        init_process_group(
            backend="nccl",       # nccl: NVIDIA Collective Communication Library
            rank=rank, 
            world_size=world_size
        )

    torch.cuda.set_device(rank)   # Sets the current GPU device on which tensors will be allocated and operations will be performed


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,                        # False because the DistributedSampler below takes care of the shuffling now
        pin_memory=True,                      # Enables faster memory transfer when training on GPU
        drop_last=True,
        sampler=DistributedSampler(train_ds)  # Splits the dataset into distinct, non-overlapping subsets for each process (GPU)
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


def main(rank, world_size, num_epochs): # The main function running the model training

    ddp_setup(rank, world_size) # Initialize DDP devices

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # Wrap model with DDP

    for epoch in range(num_epochs):
        # Set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # rank is the GPU ID
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad() # Set gradients of previous round to 0 to prevent unintented gradient accumulation
            loss.backward()       # Computes the gradients of the loss given the model parameters
            optimizer.step()      # Optimizer uses the gradients to update the model parameters

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy:", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy:", test_acc)

    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP_Multiprocessing_Test.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, by uncommenting the code on lines 106 to 109."
        )

    destroy_process_group()  # Cleanly exit distributed mode and release any resources used


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # This script may not work for GPUs > 2 due to the small dataset
    # Run `CUDA_VISIBLE_DEVICES=0,1 python DDP_Multiprocessing_Test.py` if you have GPUs > 2
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    num_epochs = 3
    world_size = torch.cuda.device_count()
    # Lauches the main function using multiple processes, where nprocs=world_size means one process per GPU
    # Spawn will automatically pass the rank
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)