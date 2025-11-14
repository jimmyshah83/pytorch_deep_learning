"""Neural network model for XOR operation."""
from torch import nn


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


if __name__ == "__main__":
    main()
