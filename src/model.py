# src/model.py

import torchvision

def create_resnet50(num_classes=1000):
    """
    Creates a ResNet-50 model.
    weights=None ensures we are training from scratch, as per project requirements.
    """
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    return model