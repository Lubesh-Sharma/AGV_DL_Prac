import random

import numpy as np

import torch
import torch.nn.functional as F


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.gard.           #
    ##############################################################################
    # Perform a forward pass through the model to get the scores
    scores = model(X)

    # Select the scores for the correct class
    correct_scores = scores[torch.arange(scores.size(0)), y]

    # Compute the loss over the correct scores
    loss = correct_scores.sum()

    # Compute the gradients with torch.autograd.grad
    gradients = torch.autograd.grad(loss, X)[0]

    # Take the absolute value of the gradients to get the saliency maps
    saliency = gradients.abs()
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image.
    X_fooling = X.clone().detach().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Set the number of iterations for gradient ascent
    num_iterations = 100

    for i in range(num_iterations):
        # Forward pass to get the scores
        scores = model(X_fooling)

        # Get the score for the target class
        target_score = scores[0, target_y]

        # Backward pass to compute gradients
        target_score.backward()

        # Get the gradient of the input image
        gradient = X_fooling.grad.data

        # Normalize the gradient
        gradient /= torch.norm(gradient)

        # Update the fooling image using gradient ascent
        X_fooling.data += learning_rate * gradient

        # Reset the gradient for the next iteration
        X_fooling.grad.data.zero_()

        # Print progress
        if (i + 1) % 10 == 0:
            print(f'Iteration {i + 1}/{num_iterations}, Target Score: {target_score.item()}')
    ##############################################################################
    return X_fooling.detach()


def update_class_visulization(model, target_y, l2_reg, learning_rate, img):
    """
    Perform one step of update on a image to maximize the score of target_y
    under a pretrained model.

    Inputs:
    - model: A pretrained CNN that will be used to generate the image
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - img: the image tensor (1, C, H, W) to start from
    """

    # Create a copy of image tensor with gradient support
    img = img.clone().detach().requires_grad_(True)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # Use the model to compute the gradient of the score for the target_y class
    scores = model(img)
    score_target_y = scores[0, target_y]
    score_target_y.backward()

    # Get the gradient of the image
    gradient = img.grad.data

    # Apply L2 regularization
    gradient += 2 * l2_reg * img.data

    # Update the image using gradient descent
    img.data += learning_rate * gradient

    # Clear the gradient for the next iteration
    img.grad.data.zero_()
    ########################################################################
    return img.detach()
