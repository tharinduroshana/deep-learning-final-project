import copy
import torch.nn as nn
import torch
from torchvision import models

pre_trained_models = {
        "resnet18": models.resnet18(pretrained=True),
        "resnet34": models.resnet34(pretrained=True),
        "vgg": models.vgg16(pretrained=True),
        "efficientnet_b0": models.efficientnet_b0(pretrained=True),
        "densenet121": models.densenet121(pretrained=True),
}

class DualImageModel(nn.Module):
    def __init__(self, model, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = pre_trained_models[model]
        backbone.fc = nn.Identity()

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        if model not in ["resnet18", "resnet34"]:
            input_features = 1000
        else:
            input_features = 512

        self.fc = nn.Sequential(
            nn.Linear(input_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
