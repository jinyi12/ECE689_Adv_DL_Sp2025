import torch
import torchvision
import torchvision.transforms as transforms


# transformations for images, similar to the previous question, except we need to discretize the pixel values
def _discretize(x):
    """
    Discretize the pixel values to 0-255.
    Need a function to be called, for pickling to work when using num_workers in pytorch lightning dataloader.
    """
    return (x * 255).to(torch.long)


def _binarize(x):
    """
    Binarize the pixel values to -1 and 1.
    """
    return torch.where(x > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
