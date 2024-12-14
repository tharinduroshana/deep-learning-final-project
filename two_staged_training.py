import pandas as pd
import os
import copy
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from template_code import transform_train, transform_test, train_model, evaluate_model, RetinopathyDataset

batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 20

class FirstStageModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
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

class SecondStageModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet18(pretrained=False)

        state_dict1 = torch.load('./trained_models_b/resnet18/first_stage_resnet18.pth', map_location='cpu')
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict1.items() if not k.startswith("fc.")}
        backbone.load_state_dict(new_state_dict, strict=False)

        backbone.fc = nn.Identity()

        for param in backbone.parameters():
            param.requires_grad = False

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)
        self.backbone3 = copy.deepcopy(backbone)
        self.backbone4 = copy.deepcopy(backbone)

        self.fc = nn.Sequential(
            nn.Linear(512 * 4, 256),
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

class SecondStageModel1(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=False)

        state_dict1 = torch.load('./trained_models_b/efficientnet_b0/efficientnet_b0_first_stage.pth', map_location='cpu')
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict1.items() if not k.startswith("fc.")}
        self.backbone.load_state_dict(new_state_dict, strict=False)

        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class APTOS2019Dataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test

        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['id_code']) + '.png'
            if not self.test:
                file_info['dr_level'] = int(row['diagnosis'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

class RetinopathyDatasetPatientLevel(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test

        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item(index)

    def load_data(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        # df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix'])

        data = []
        for (prefix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            file_info['img_path3'] = os.path.join(self.image_dir, group.iloc[2]['img_path'])
            file_info['img_path4'] = os.path.join(self.image_dir, group.iloc[3]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')
        img3 = Image.open(data['img_path3']).convert('RGB')
        img4 = Image.open(data['img_path4']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2, img3, img4], label
        else:
            return [img1, img2, img3, img4]

if __name__ == '__main__':
    first_stage = False

    if first_stage:
        model = FirstStageModel()

        train_dataset = APTOS2019Dataset('./APTOS2019/train_1.csv', './APTOS2019/train_images/train_images/', transform_train)
        val_dataset = APTOS2019Dataset('./APTOS2019/valid.csv', './APTOS2019/val_images/val_images/', transform_test)
        test_dataset = APTOS2019Dataset('./APTOS2019/test.csv', './APTOS2019/test_images/test_images/', transform_test, test=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # Use GPU device is possible
        device = torch.device('mps')
        print('Device:', device)

        # Move class weights to the device
        model = model.to(device)

        # Optimizer and Learning rate scheduler
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train and evaluate the model with the training and validation set
        model = train_model(
            model, train_loader, val_loader, device, criterion, optimizer,
            lr_scheduler=lr_scheduler, num_epochs=num_epochs,
            checkpoint_path='./efficientnet_b0_first_stage.pth'
        )

        # Load the pretrained checkpoint
        state_dict = torch.load('./efficientnet_b0_first_stage.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

        # Make predictions on testing set and save the prediction results
        evaluate_model(model, test_loader, device, test_only=True)
    else:
        # model = SecondStageModel1()
        model = SecondStageModel()

        # train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train)
        # val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
        # test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, test=True)

        train_dataset = RetinopathyDatasetPatientLevel('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train)
        val_dataset = RetinopathyDatasetPatientLevel('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
        test_dataset = RetinopathyDatasetPatientLevel('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, test=True)

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

        # Train and evaluate the model with the training and validation set
        model = train_model(
            model, train_loader, val_loader, device, criterion, optimizer,
            lr_scheduler=lr_scheduler, num_epochs=num_epochs,
            checkpoint_path='./second_stage_resnet18_pt_level.pth'
        )

        # Load the pretrained checkpoint
        state_dict = torch.load('./second_stage_resnet18_pt_level.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

        # Make predictions on testing set and save the prediction results
        evaluate_model(model, test_loader, device, test_only=True)