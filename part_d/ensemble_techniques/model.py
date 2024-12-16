import copy
import torch.nn as nn
import torch

class SecondStageModel(nn.Module):
    def __init__(self, model, in_features, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = model

        backbone.fc = nn.Identity()

        for param in backbone.parameters():
            param.requires_grad = False

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)
        self.backbone3 = copy.deepcopy(backbone)
        self.backbone4 = copy.deepcopy(backbone)

        self.fc = nn.Sequential(
            nn.Linear(in_features * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2, image3, image4 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)
        x3 = self.backbone3(image3)
        x4 = self.backbone4(image4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc(x)
        return x
