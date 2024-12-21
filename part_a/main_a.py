import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from part_a.dataset import RetinopathyDataset
from part_a.dual_image_model import DualImageModel
from part_a.single_image_model import SingleImageModel
from shared.shared import select_device, evaluate_model, train_model, \
    transform_train, transform_test, batch_size, num_epochs, learning_rate

if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    # This will affect the model definition, dataset pipeline, training and evaluation

    mode = 'single'  # forward single image to the model each time
    # mode = 'dual'  # forward two images of the same eye to the model and fuse the features

    # model_type = 'resnet18'
    # model_type = 'resnet34'
    # model_type = 'vgg16'
    # model_type = 'efficientnet_b0'
    model_type = 'densenet121'

    assert mode in ('single', 'dual')
    assert model_type in ('resnet18', 'resnet34', 'vgg16', 'efficientnet_b0', 'densenet121')

    # Define the model
    if mode == 'single':
        model = SingleImageModel(model_type)
    else:
        model = DualImageModel(model_type)

    print('Pipeline Mode:', mode)

    # Use GPU device is possible
    device = select_device()
    print('Device:', device)

    # Create datasets
    train_dataset = RetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('../DeepDRiD/test.csv', '../DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move class weights to the device
    model = model.to(device)

    # Optimizer and Learning rate scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = './outputs/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train and evaluate the model with the training and validation set
    model = train_model(
        model, train_loader, val_loader, device, criterion, optimizer,
        lr_scheduler=lr_scheduler, num_epochs=num_epochs,
        checkpoint_path=f'outputs/{model_type.lower()}_{mode.lower()}_mode.pth',
        visualizations_save_path=f'outputs/{model_type.lower()}_accuracy_and_loss.png'
    )

    # Load the pretrained checkpoint
    state_dict = torch.load(f'outputs/{model_type.lower()}_{mode.lower()}_mode.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # Make predictions on testing set and save the prediction results
    evaluate_model(model, test_loader, device, test_only=True, prediction_path='outputs/test_predictions.csv')
