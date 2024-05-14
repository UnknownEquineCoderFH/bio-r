import torch as tch
from torch import nn
from matplotlib import pyplot as plt
from itertools import islice, tee
from typing import Iterable, TypeVar, Iterator

T = TypeVar("T")  # PY3.11 generics syntax


# Given an iterable, this function returns a sliding window of the given size
# Ex: sliding_window([1, 2, 3, 4, 5], 3) -> [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
# It gives the SimpleNeuralNetwork api a way to create layers of the given size
def sliding_window(iterable: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    iterables = tee(iter(iterable), size)
    window = zip(*(islice(t, n, None) for n, t in enumerate(iterables)))
    yield from window


class SVM(tch.nn.Module):
    def __init__(
        self,
        input_n: int,
        output_n: int,
        weights_size: tuple[int, ...],
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Variable number of layers, based on the provides sizes
        layers = list[tch.nn.Linear]()
        sizes = sliding_window([input_n, *weights_size, output_n], 2)

        for in_size, out_size, *_ in sizes:
            layer = tch.nn.Linear(in_size, out_size)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: tch.Tensor) -> tch.Tensor:
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                x = tch.nn.functional.softmax(layer(x))
            else:
                # One point of improvement: Maybe relu is not the best activation function
                x = tch.nn.functional.relu(layer(x))

        return x


def tch_train_model(
    X_train: tch.Tensor,
    y_train: tch.Tensor,
    loss_function: tch.nn.Module,
    size: tuple[int, ...],
    eta: float,
    epochs: int,
    verbose: bool = False,
) -> tuple[SVM, list[float], list[float]]:
    losses = list[float]()
    accuracies = list[float]()

    model = SVM(X_train.shape[1], len(y_train.unique()), size)

    optimiser = tch.optim.Adam(model.parameters(), lr=eta)

    for epoch in range(1, epochs + 1):
        # Zero the gradients
        optimiser.zero_grad()

        # Forward pass
        output = model(X_train)

        # Calculate the loss
        loss = loss_function(output, y_train)

        # Backward pass
        loss.backward()

        # Update the weights
        optimiser.step()

        # Log the loss
        if epoch % 25 == 0:
            if verbose:
                print(f"Epoch {epoch} loss: {loss.item()}")

            losses.append(loss.item())
            accuracies.append(
                (tch.argmax(output, dim=1) == y_train).sum().item() / len(y_train)
            )

    return model, losses, accuracies


def tch_evaluate_model(model: SVM, X_test: tch.Tensor, y_test: tch.Tensor) -> float:
    with tch.no_grad():
        # Get the predictions
        predictions = model(X_test)

        # Get the predicted classes
        predicted_classes = tch.argmax(predictions, dim=1)

        # Calculate the accuracy
        accuracy = (predicted_classes == y_test).sum().item() / len(y_test)

    return accuracy


def tch_plot_model(
    losses: list[float],
    accuracies: list[float],
    epochs: int,
    size: tuple[int, ...],
    eta: float,
    loss_function: tch.nn.Module,
    framework: str,
) -> None:
    # Plot the losses
    plt.plot(range(1, epochs + 1, int(epochs / len(losses))), losses)
    plt.title(
        f"{framework} Neural Network Loss with layers of size {size},"
        f" learning rate {eta} and loss function {loss_function}"
    )
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

    # Plot the accuracies
    plt.plot(range(1, epochs + 1, int(epochs / len(losses))), accuracies)
    plt.title(
        f"{framework} Neural Network Accuracy with layers of size {size}"
        f" learning rate {eta} and loss function {loss_function}"
    )
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy (%)")
    plt.show()


def tch_pipeline(
    size: tuple[int, ...],
    eta: float,
    epochs: int,
    X_train: tch.Tensor,
    y_train: tch.Tensor,
    X_test: tch.Tensor,
    y_test: tch.Tensor,
    loss_function: tch.nn.Module,
    framework: str = "PyTorch",
) -> None:
    model, losses, accuracies = tch_train_model(
        X_train, y_train, loss_function, size, eta, epochs
    )

    accuracy = tch_evaluate_model(model, X_test, y_test)

    print(f"Accuracy on training set: {accuracies[-1] * 100}%")
    print(f"Accuracy on test set: {accuracy * 100}%")

    tch_plot_model(losses, accuracies, epochs, size, eta, loss_function, framework)


sizes: list[tuple[int, ...]] = [
    (16, 32),  # One hidden layer
    (64, 128),  # One Hidden Layer
    (
        32,
        32,
        32,
        32,
    ),  # Three Hidden Layers [(IN, 32), (32, 32), (32, 32), (32, 32), (32, OUT)]
    (64, 128, 128, 64),  # Three Hidden Layers
    (24, 48, 192, 48, 24),  # Four Hidden Layers
]

lrs = [0.0003, 0.003]  # Learning rates

loss_functions: list[nn.Module] = [
    tch.nn.CrossEntropyLoss(),
    tch.nn.NLLLoss(),
]  # Tinkering with loss functions


# Here itertools.product is sadly not the best choice because we lose type information
tch_parameters = [
    (size, lr, loss_function)
    for size in sizes
    for lr in lrs
    for loss_function in loss_functions
]
