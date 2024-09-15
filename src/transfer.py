import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    """
    Loads a pre-trained model, freezes the parameters, and replaces the final fully connected layer 
    for the specified number of classes.
    
    :param model_name: Name of the model architecture (default: 'resnet18')
    :param n_classes: Number of output classes for the new task (default: 50)
    :return: Modified model with the final layer replaced for transfer learning
    """
    # Get the requested architecture from torchvision models
    if hasattr(models, model_name):
        # Ensure correct capitalization of the model name to access the correct weights
        model_name_camel = f"ResNet18" if model_name == "resnet18" else model_name.capitalize()
        weights_attr = f"{model_name_camel}_Weights"
        weights = getattr(models, weights_attr).IMAGENET1K_V1
        model_transfer = getattr(models, model_name)(weights=weights)
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Get the number of input features to the final fully connected layer
    num_ftrs = model_transfer.fc.in_features

    # Replace the final fully connected layer with a new layer
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):
    """
    Test if the transfer learning model is working as expected.
    Checks the output tensor's shape and verifies that the model is correctly adapted 
    for the specified number of output classes.
    """

    # Get the model with 23 output classes
    model = get_model_transfer_learning(n_classes=23)

    # Get a batch of images from the data loader
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Run the model on the batch of images
    out = model(images)

    # Ensure the output is a tensor and matches the batch size and number of classes
    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
