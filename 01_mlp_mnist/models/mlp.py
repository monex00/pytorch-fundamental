import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected classifier for MNIST-like inputs."""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout_p=0.3):
        super().__init__()
        layers = []

        # Input comes as (N, C, H, W); Flatten makes it (N, input_size).

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(inplace=True),
                # Normalize activations to stabilize training.
                nn.BatchNorm1d(hidden_sizes[0]),
                # Regularize to reduce overfitting.
                nn.Dropout(dropout_p),
            ]
        )

        for i in range(1, len(hidden_sizes)):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(hidden_sizes[i]),
                    nn.Dropout(dropout_p),
                ]
            )

        # Final linear layer outputs raw class logits.
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Returns logits; apply softmax only for inference/visualization.
        out = self.network(x)
        return out
