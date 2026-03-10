import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Loading VGG19 pre-trained on ImageNet
        vgg = models.vgg19(pretrained=True).features

        self.content_layers = ["conv4_2"]
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        # Mapping name -> index in the VGG layers
        self.layer_indices = {
            "conv1_1": 0,
            "conv2_1": 5,
            "conv3_1": 10,
            "conv4_1": 19,
            "conv4_2": 21,
            "conv5_1": 28,
        }

        # Layers we want to extract
        self.wanted_layers = set(self.content_layers + self.style_layers)

        # Saving the VGG layers
        self.vgg_layers = vgg

        # Freeze the parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}

        # Pass through VGG layer by layer
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)

            # Check if this index corresponds to a layer we want
            for name, idx in self.layer_indices.items():
                if i == idx and name in self.wanted_layers:
                    features[name] = x

        return features
