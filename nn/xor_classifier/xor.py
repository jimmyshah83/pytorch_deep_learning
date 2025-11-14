"""Neural network model for XOR operation."""
from torch import nn
import torch
from torch.utils import data


class XORDataset(data.Dataset):    
    """Dataset for XOR operation."""
    def __init__(self, size, std=0.1):
        """Initialize the dataset."""
        self.size = size
        self.std = std
        self.generate_continuous_xor()
        
    def generate_continuous_xor(self):
        """Generate continuous XOR data."""
        x_data = torch.randint(0, 2, (self.size, 2), dtype=torch.float32)
        label = (x_data.sum(dim=1) == 1).to(torch.long)
        # Add noise to the data
        x_data += torch.randn(x_data.shape) * self.std
        self.data = x_data
        self.label = label
        
    def __len__(self):
        """Return the length of the dataset."""
        return self.size
    
    def __getitem__(self, index):
        """Return the data and label at the given index."""
        return self.data[index], self.label[index]


class XORModel(nn.Module):
    """Neural network model for XOR operation."""
    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Initialize the model."""
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        """Forward pass of the model."""
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        return x


def main():
    """Main function to run the XOR model."""
    model = XORModel(num_inputs=2, num_hidden=4, num_outputs=1)
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter: {name}:, Shape: {param.shape}")
        
    dataset = XORDataset(size=200)
    print(f"Data: {dataset[0][0]}, Label: {dataset[0][1]}")
    
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)
    data_inputs, data_labels = next(iter(dataloader))
    print(f"Data inputs: {data_inputs}, Data labels: {data_labels}")
    
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == "__main__":
    main()
