import torch
import torch.nn as nn
import torchvision.models as models

class RGBDResNetRegression(nn.Module):
    def __init__(self, base_model='resnet18', pretrained=True,
                 in_channels=4, kernel_size=3):
        super(RGBDResNetRegression, self).__init__()
        # Load a pre-trained ResNet model
        if base_model == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif base_model == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif base_model == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported base model")

        # Modify the first convolutional layer to accept 4 channels (RGBD)
        if in_channels == 4:
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False)
        else: # use the default 3 channels (RGB)
            pass

        # Feature extractor part of ResNet
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # Add a ReLU before the final linear layer
        self.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.ReLU()  # Ensure the output is non-negative
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten the tensor
        x = self.fc(x)
        return x