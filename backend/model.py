
import torch
import torch.nn as nn
from torchvision import models


# Select GPU if available, otherwise use CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Number of Food-101 categories
NUM_CLASSES = 101


# Load the model and its preprocessing transforms
def load_model():

    # Get the default ConvNeXt Small weights configuration
    weights = models.ConvNeXt_Small_Weights.DEFAULT

    # Create ConvNeXt Small architecture
    model = models.convnext_small(weights=None)

    # Replace the original classifier with our 101-class classifier
    model.classifier = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=768, out_features=NUM_CLASSES)
    )

    # Load the trained checkpoint
    checkpoint_path = "models/convnext_final.pth"

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Handle checkpoints saved from torch.compile(), if necessary
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    # Move model to GPU or CPU
    model = model.to(DEVICE)

    # Set model to evaluation mode
    model.eval()

    # Get the correct preprocessing transformations
    transform = weights.transforms()

    return model, transform
