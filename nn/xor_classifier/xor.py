"""Neural network model for XOR operation."""
from torch import nn
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


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
                device=torch.device("cpu"), optimizer=None, writer=None):
    """Train the model."""
    model.train()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.
        num_batches = 0.
        
        for batch_idx, (data_inputs, data_labels) in enumerate(dataloader):
            # Step 1: Move input data to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            # Step 2: Run the model on the input data
            predictions = model(data_inputs)
            # Output is [Batch size, 1], but we want [Batch size]
            predictions = predictions.squeeze(dim=1)
            # Step 3: Calculate the loss
            loss = loss_function(predictions, data_labels.float())
            # Step 4: Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Step 5: Update the parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log batch loss to TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        
        # Log epoch loss to TensorBoard
        if writer is not None:
            writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
            
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss}")


def load_model(model, path):
    """Load the model."""
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


def evaluate_model(model, dataloader, device=torch.device("cpu"),
                   writer=None, epoch=None):
    """Evaluate the model."""
    model.eval()
    true_predictions = 0.
    total_predictions = 0.
    
    with torch.no_grad():
        for data_inputs, data_labels in dataloader:
            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            predictions = model(data_inputs)
            predictions = predictions.squeeze(dim=1)
            # Sigmoid to map predictions between 0 and 1
            predictions = torch.sigmoid(predictions)
            prediction_labels = (predictions > 0.5).long()
            
            # Keep records of predictions for accuracy metric
            # (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_predictions += (prediction_labels == data_labels).sum()
            total_predictions += data_labels.shape[0]
    
    accuracy = true_predictions / total_predictions
    
    # Log accuracy to TensorBoard
    if writer is not None:
        step = epoch if epoch is not None else 0
        writer.add_scalar('Eval/Accuracy', accuracy, step)
    
    print(f"Accuracy: {accuracy}")
    return accuracy


def main():
    """Main function to run the XOR model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XORModel(num_inputs=2, num_hidden=4, num_outputs=1).to(device)
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter: {name}:, Shape: {param.shape}")
    
    # TensorBoard Setup
    writer = SummaryWriter('runs/xor_classifier')
    
    # Log model architecture
    train_dataset = XORDataset(size=2500)
    sample_loader = data.DataLoader(train_dataset, batch_size=1)
    sample_input = next(iter(sample_loader))[0].to(device)
    writer.add_graph(model, sample_input)

    # Training
    dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Loss Function
    loss_function = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Log hyperparameters
    writer.add_hparams(
        {'lr': 0.1, 'batch_size': 128, 'num_epochs': 10, 'hidden_size': 4},
        {}
    )
    
    model.to(device)
    train_model(model=model, dataloader=dataloader,
                loss_function=loss_function, device=device,
                optimizer=optimizer, num_epochs=10, writer=writer)

    # Saving the model
    state_dict = model.state_dict()
    print(f"State dict: {state_dict}")
    
    torch.save(state_dict, "xor_model.pth")

    # Evaluating the model
    test_dataset = XORDataset(size=500)
    test_data_loader = data.DataLoader(test_dataset, batch_size=128,
                                        shuffle=False, drop_last=False)
    evaluate_model(model=model, dataloader=test_data_loader, device=device,
                   writer=writer, epoch=10)
    
    # Log model weights histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, 10)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, 10)
    
    # Close TensorBoard writer
    writer.close()
    print("TensorBoard logs saved to 'runs/xor_classifier'.")
    print("Run 'tensorboard --logdir=runs' to view.")
    
    
if __name__ == "__main__":
    main()
