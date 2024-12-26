import torch
from torch import nn
import matplotlib.pyplot as plt

class ModelLinearFunction(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  * self.weight + self.bias


def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.savefig("chart")

# prepare data
weight = 0.5
bias = 0.15
x_original = torch.arange(0, 2, 0.15).unsqueeze(dim=1)
y_original = x_original * weight + bias

# split data into two chunks for_training and for_test
limit = int(len(x_original) * 0.8)
x_for_training = x_original[:limit]
y_for_training = y_original[:limit]

x_for_testing = x_original[limit:]
y_for_testing = y_original[limit:]

# dump chunks
# print(f"len(x_for_training): {len(x_for_training)}\n\n x_for_training: {x_for_training}\n\n y_for_training: {y_for_training}\n\n len(x_for_testing): {len(x_for_testing)}\n\n x_for_testing: {x_for_testing}\n\n y_for_testing: {y_for_testing}")

# setup training
torch.manual_seed(18)
model = ModelLinearFunction()
loss_fn = nn.L1Loss()
optimizer_fn = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 200

# training loop
for epoch in range(epochs):
    model.train()
    y_prediction = model(x_for_training)
    loss = loss_fn(y_prediction, y_for_training)
    optimizer_fn.zero_grad()
    loss.backward()
    optimizer_fn.step()

    #Testing
    model.eval()
    with torch.inference_mode():
        y_test_pred = model(x_for_testing)
        test_loss = loss_fn(y_test_pred, y_for_testing)

    if epoch % 10 == 0:
        print("Epoch: {}, loss: {}, test_loss: {}".format(epoch, loss, test_loss))

print(f"Expected: weight: {weight}, bias: {bias}. Got model.state_dict(): {model.state_dict()}")
plot_predictions(x_for_training, y_for_training, x_for_testing, y_for_testing, model(x_for_testing).detach().numpy())
