import torch
import torch.nn.functional as F


def gram_matrix(features):
    """
    Calcola la Gram matrix per catturare le correlazioni tra feature maps.

    Args:
        features: tensor (B, C, H, W)

    Returns:
        gram: tensor (B, C, C)
    """
    shape = features.shape
    B, C, H, W = shape
    F_flat = features.view(B, C, H * W)
    # Gram: (256, 4096) @ (4096, 256) = (256, 256)
    G = torch.matmul(F_flat, F_flat.transpose(1, 2))
    #G = G / (C * H * W)
    return G


def content_loss(gen_features, content_features):
    """
    MSE tra le feature del contenuto.

    Args:
        gen_features: dict {layer_name: tensor}
        content_features: dict {layer_name: tensor}

    Returns:
        loss: scalar tensor
    """
    layer = "conv4_2"
    loss = F.mse_loss(gen_features[layer], content_features[layer])
    return loss


def style_loss(gen_features, style_features, style_layers):
    """
    MSE tra le Gram matrices.

    Args:
        gen_features: dict {layer_name: tensor}
        style_features: dict {layer_name: tensor}
        style_layers: list di layer names

    Returns:
        loss: scalar tensor
    """
    loss = 0
    for layer in style_layers:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        layer_loss = F.mse_loss(gen_gram, style_gram)
        loss += layer_loss

    return loss
