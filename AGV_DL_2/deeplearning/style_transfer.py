import numpy as np

import torch
import torch.nn.functional as F


def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    # Compute the mean squared error between content_current and content_target
    loss_content = F.mse_loss(content_current, content_target)

    # Scale the content loss by the content_weight
    loss_content *= content_weight

    return loss_content


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.size()

    # Reshape the features to be of shape (N, C, H*W)
    features_reshaped = features.view(N, C, H * W)

    # Compute the Gram matrix by matrix multiplication
    gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))

    # Optionally normalize the Gram matrix
    if normalize:
        gram /= (H * W * C)

    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    ##############################################################################
    style_loss = 0

    # Iterate over the style layers
    for i, layer in enumerate(style_layers):
        # Compute the Gram matrix for the current image at the specified layer
        current_feats = feats[layer]
        current_gram = gram_matrix(current_feats)

        # Compute the style loss at the current layer
        layer_loss = style_weights[i] * F.mse_loss(current_gram, style_targets[i])

        # Accumulate the style loss
        style_loss += layer_loss

    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    ##############################################################################
    # Compute differences along height and width
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]

    # Compute the total variation loss
    loss = tv_weight * (torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw)))

    return loss
