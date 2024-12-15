import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F

from enum import Enum

from template_code import RetinopathyDataset, transform_train, transform_test, train_model, evaluate_model

batch_size = 24
num_classes = 5
learning_rate = 0.0001
num_epochs = 20

TRAINING_MODE = 'dual'


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.fc(x)
        attention = self.sigmoid(attention)
        return x * attention


class MyDualModelWithChannelAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.densenet121(pretrained=True)
        backbone.fc = nn.Identity()

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.backbone1(dummy_input)

        self.num_channels = dummy_output.size(1)

        self.channel_attention1 = ChannelAttention(in_channels=self.num_channels)
        self.channel_attention2 = ChannelAttention(in_channels=self.num_channels)

        self.fc = nn.Sequential(
            nn.Linear(self.num_channels * 2, 256),
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

        x1 = self.channel_attention1(x1)
        x2 = self.channel_attention2(x2)

        x = torch.cat((x1, x2), dim=1)

        x = self.fc(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class MyDualModelWithSpatialAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.densenet121(pretrained=True)
        backbone.fc = nn.Identity()

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.backbone1(dummy_input)

        self.num_channels = dummy_output.size(1)

        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        self.spatial_attention2 = SpatialAttention(kernel_size=7)

        self.fc = nn.Sequential(
            nn.Linear(1024 * 2 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = torch.squeeze(image1, dim=0)
        x2 = torch.squeeze(image2, dim=0)

        x1 = self.backbone1.features(x1)
        x2 = self.backbone2.features(x2)

        x1 = self.spatial_attention1(x1)
        x2 = self.spatial_attention2(x2)

        x = torch.cat((x1, x2), dim=1)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_output = torch.matmul(attention_weights, V)
        return attended_output


class MyDualModelWithSelfAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.densenet121(pretrained=True)
        backbone.fc = nn.Identity()

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        self.self_attention1 = SelfAttention(1000)
        self.self_attention2 = SelfAttention(1000)

        self.fc = nn.Sequential(
            nn.Linear(1000 * 2, 256),
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

        x1 = self.self_attention1(x1)
        x2 = self.self_attention2(x2)

        x = torch.cat((x1, x2), dim=1)

        x = self.fc(x)
        return x


class AttentionModes(Enum):
    SELF = MyDualModelWithSelfAttention
    CHANNEL = MyDualModelWithChannelAttention
    SPATIAL = MyDualModelWithSpatialAttention


if __name__ == '__main__':
    mode = TRAINING_MODE
    attention_mode = AttentionModes.CHANNEL

    model = attention_mode.value()

    print('Pipeline Mode:', mode)

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = torch.device('mps')
    print('Device:', device)

    # Move class weights to the device
    model = model.to(device)

    # Optimizer and Learning rate scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint_dir = '.artifacts/task_c/densenet121/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train and evaluate the model with the training and validation set
    model = train_model(
        model, train_loader, val_loader, device, criterion, optimizer,
        lr_scheduler=lr_scheduler, num_epochs=num_epochs,
        checkpoint_path=f'.artifacts/task_c/densenet121/densenet121_{attention_mode.name.lower()}_attention.pth'
    )

    # Load the pretrained checkpoint
    state_dict = torch.load(f'.artifacts/task_c/densenet121/densenet121_{attention_mode.name.lower()}_attention.pth',
                            map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # Make predictions on testing set and save the prediction results
    evaluate_model(model, test_loader, device, test_only=True)
