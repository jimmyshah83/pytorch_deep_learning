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


def train_model(model, dataloader, loss_function, num_epochs=100, 
                device=torch.device("cpu"), optimizer=None):
    """Train the model."""
    model.train()
    
    # Training loop
    for epoch in range(num_epochs):
        for data_inputs, data_labels in dataloader:
             ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
             ## Step 2: Run the model on the input data
            predictions = model(data_inputs)
            # Output is [Batch size, 1], but we want [Batch size]
            predictions = predictions.squeeze(dim=1) 
            ## Step 3: Calculate the loss
            loss = loss_function(predictions, data_labels.float())
            ## Step 4: Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()


def load_model(model, path):
    """Load the model."""
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


def main():
    """Main function to run the XOR model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XORModel(num_inputs=2, num_hidden=4, num_outputs=1).to(device)
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter: {name}:, Shape: {param.shape}")
    
    ############ Training ############
    train_dataset = XORDataset(size=2500)
    dataloader = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    ############ Loss Function ############
    loss_function = nn.BCEWithLogitsLoss()
    
    ############ Optimizer ############
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    model.to(device)
    train_model(model=model, dataloader=dataloader, 
                loss_function=loss_function, device=device, 
                optimizer=optimizer, num_epochs=10)
    
    ##### Saving the model #####
    state_dict = model.state_dict()
    print(f"State dict: {state_dict}")
    
    torch.save(state_dict, "xor_model.pth")
    
    
if __name__ == "__main__":
    main()
