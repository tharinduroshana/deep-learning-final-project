from enum import Enum
from torchvision import models
import torch.nn as nn
import torch
import copy
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from shared.shared import transform_train, transform_test, train_model, evaluate_model, RetinopathyDataset, \
    select_device, batch_size, learning_rate, num_epochs

MODEL_TYPES = {
    "VGG16": models.vgg16,
    "RESNET18": models.resnet18,
    "RESNET34": models.resnet34,
    "DENSENET121": models.densenet121,
    "EFFICIENTNET_B0": models.efficientnet_b0
}


class TrainingModes(Enum):
    STANDARD = "standard"
    PATIENT_LEVEL = "patient_level"


def _get_input_features(model_type: str):
    if model_type not in ["RESNET18", "RESNET34"]:
        return 1000
    else:
        return 512


class FirstStageModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5, model_type: str = "RESNET18"):
        super().__init__()

        self.backbone = MODEL_TYPES[model_type](pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        input_features = _get_input_features(model_type=model_type)
        print("Input features:", input_features)

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


class SecondStageModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5, model_type: str = "RESNET18",
                 training_mode: TrainingModes = TrainingModes.STANDARD):
        super().__init__()

        backbone = MODEL_TYPES[model_type](pretrained=False)

        input_features = _get_input_features(model_type)

        print("Input features:", input_features)

        state_dict1 = torch.load(
            f'.artifacts/task_b/{model_type.lower()}/first_stage_{model_type.lower()}.pth',
            map_location='cpu', weights_only=False)
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict1.items() if not k.startswith("fc.")}
        backbone.load_state_dict(new_state_dict, strict=False)

        backbone.fc = nn.Identity()

        for param in backbone.parameters():
            param.requires_grad = False

        if training_mode == TrainingModes.PATIENT_LEVEL.value:
            # Here the two backbones will have the same structure but unshared weights
            self.backbone1 = copy.deepcopy(backbone)
            self.backbone2 = copy.deepcopy(backbone)
            self.backbone3 = copy.deepcopy(backbone)
            self.backbone4 = copy.deepcopy(backbone)
            self.image_count = 4
        else:
            self.backbone = copy.deepcopy(backbone)
            self.image_count = 1

        self.fc = nn.Sequential(
            nn.Linear(input_features * self.image_count, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, images):
        if self.image_count == 4:
            image1, image2, image3, image4 = images

            x1 = self.backbone1(image1)
            x2 = self.backbone2(image2)
            x3 = self.backbone3(image3)
            x4 = self.backbone4(image4)

            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            x = self.backbone(images)

        x = self.fc(x)
        return x

    def load_pretrained_backbone(self, path):
        pretrained_state_dict = torch.load(path)
        self.backbone.load_state_dict(pretrained_state_dict)


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


def train_first_stage_model(model_type: str):
    print(model_type)
    first_stage_model = FirstStageModel(model_type=model_type)

    train_dataset = APTOS2019Dataset('../APTOS2019/train_1.csv', '../APTOS2019/train_images/train_images/',
                                     transform_train)
    val_dataset = APTOS2019Dataset('../APTOS2019/valid.csv', '../APTOS2019/val_images/val_images/',
                                   transform_test)
    test_dataset = APTOS2019Dataset('../APTOS2019/test.csv', '../APTOS2019/test_images/test_images/',
                                    transform_test, test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = select_device()
    print('Device:', device)

    # Move class weights to the device
    first_stage_model = first_stage_model.to(device)

    optimizer = torch.optim.Adam(params=first_stage_model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint_dir = f'.artifacts/task_b/{model_type.lower()}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    first_stage_model = train_model(
        first_stage_model, train_loader, val_loader, device, criterion, optimizer,
        lr_scheduler=lr_scheduler, num_epochs=num_epochs,
        checkpoint_path=f'.artifacts/task_b/{model_type.lower()}/first_stage_{model_type.lower()}.pth'
    )

    # Load the pretrained checkpoint
    state_dict = torch.load(f'.artifacts/task_b/{model_type.lower()}/first_stage_{model_type.lower()}.pth',
                            map_location='cpu', weights_only=False)
    first_stage_model.load_state_dict(state_dict, strict=True)

    # Make predictions on testing set and save the prediction results
    evaluate_model(first_stage_model, test_loader, device, test_only=True)


def train_second_stage_model(model_type, training_mode):
    print(model_type, training_mode)
    model = SecondStageModel(model_type=model_type, training_mode=training_mode)

    if training_mode == TrainingModes.STANDARD.value:
        train_dataset = RetinopathyDataset('../DeepDRiD/train.csv', './DeepDRiD/train/', transform_train)
        val_dataset = RetinopathyDataset('../DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
        test_dataset = RetinopathyDataset('../DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, test=True)
    else:
        train_dataset = RetinopathyDatasetPatientLevel('../DeepDRiD/train.csv', './DeepDRiD/train/', transform_train)
        val_dataset = RetinopathyDatasetPatientLevel('../DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
        test_dataset = RetinopathyDatasetPatientLevel('../DeepDRiD/test.csv', './DeepDRiD/test/', transform_test,
                                                      test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = select_device()
    print('Device:', device)

    # Move class weights to the device
    model = model.to(device)

    # Optimizer and Learning rate scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint_dir = f'.artifacts/task_b/{model_type.lower()}/{training_mode.lower()}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = train_model(
        model, train_loader, val_loader, device, criterion, optimizer,
        lr_scheduler=lr_scheduler, num_epochs=num_epochs,
        checkpoint_path=f'.artifacts/task_b/{model_type.lower()}/{training_mode.lower()}/second_stage_{model_type.lower()}_{training_mode.lower()}.pth'
    )

    # Load the pretrained checkpoint
    state_dict = torch.load(
        f'.artifacts/task_b/{model_type.lower()}/{training_mode.lower()}/second_stage_{model_type.lower()}_{training_mode.lower()}.pth',
        map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    # Make predictions on testing set and save the prediction results
    evaluate_model(model, test_loader, device, test_only=True)


if __name__ == '__main__':
    # Supported model types: VGG16, RESNET18, RESNET34, DENSENET121, EFFICIENTNET_B0
    model_type = "VGG16"
    training_mode = TrainingModes.STANDARD.value

    train_first_stage_model(model_type=model_type)

    train_second_stage_model(model_type=model_type, training_mode=training_mode)
