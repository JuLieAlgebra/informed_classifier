"""
Demonstration of typical ML solution when there is no information available about the system in question.

Neural network.
"""
import glob

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CustomDataset:
    def __init__(self, file_dir: str):
        # File dir should be "test", "train", "validation"
        self.filenames = glob.glob(os.path.join(file_dir, "*"))

    def __len__(self):
        return len(self.filenames)

    def __get_item__(self, idx):
        datapoint = np.load(self.filenames[idx])
        label = int(datapoint[-1])
        sample = datapoint[:-1]
        return sample, label


def test(model):
    model.eval()
    torch.set_grad_enabled(False)


def train(model, dataloader, epochs: int):
    model.train()
    torch.set_grad_enabled(True)

    for epoch in epochs:
        for datapoint in dataloader:
            torch.cuda.synchronize()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    model = NeuralNetwork().to(device)
    print(model)
