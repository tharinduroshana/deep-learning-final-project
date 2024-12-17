import torch.nn as nn
from torchvision import models

pre_trained_models = {
        "resnet18": models.resnet18(pretrained=True),
        "resnet34": models.resnet34(pretrained=True),
        "vgg16": models.vgg16(pretrained=True),
        "efficientnet_b0": models.efficientnet_b0(pretrained=True),
        "densenet121": models.densenet121(pretrained=True),
}

class SingleImageModel(nn.Module):
    def __init__(self, model, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = pre_trained_models[model]
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        if model not in ["resnet18", "resnet34"]:
            input_features = 1000
        else:
            input_features = 512

        self.fc = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
