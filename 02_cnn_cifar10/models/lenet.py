import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 32×32×3 → 28×28×6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28×28×6 → 14×14×6
            nn.Conv2d(6, 16, kernel_size=5),  # 14×14×6 → 10×10×16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10×10×16 → 5×5×16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
