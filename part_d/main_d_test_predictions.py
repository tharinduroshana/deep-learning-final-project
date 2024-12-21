import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision import models

from part_d.dataset import PatientLevelRetinopathyDataset
from part_d.image_pre_processing.image_pre_processing import circle_cropping, \
    gaussian_blur, sharpen_image
from part_d.model import SecondStageModel
from shared.shared import transform_test, select_device, batch_size, num_classes, evaluate_model


def load_saved_models(num_classes, device):
    saved_models = {
        "densenet121": models.densenet121(weights=None),
        "efficientnet_b0": models.efficientnet_b0(weights=None),
        "resnet34": models.resnet34(weights=None),
    }

    in_features = {
        "densenet121": 1000,
        "efficientnet_b0": 1000,
        "resnet34": 512,
    }

    model_densenet121 = SecondStageModel(saved_models["densenet121"], in_features["densenet121"], num_classes=num_classes).to(device)
    state_dict = torch.load('saved_models/densenet121.pth', map_location=device, weights_only=True)
    model_densenet121.load_state_dict(state_dict, strict=True)

    model_efficientnet_b0 = SecondStageModel(saved_models["efficientnet_b0"], in_features["efficientnet_b0"], num_classes=num_classes).to(device)
    state_dict = torch.load('saved_models/efficientnet_b0.pth', map_location=device, weights_only=True)
    model_efficientnet_b0.load_state_dict(state_dict, strict=True)

    model_resnet34 = SecondStageModel(saved_models["resnet34"], in_features["resnet34"], num_classes=num_classes).to(device)
    state_dict = torch.load('saved_models/resnet34.pth', map_location=device, weights_only=True)
    model_resnet34.load_state_dict(state_dict, strict=True)

    return [model_densenet121, model_efficientnet_b0, model_resnet34]

def validate_ensemble(final_predictions, val_loader):
    true_labels = [labels.numpy() for _, labels in val_loader]
    true_labels = np.concatenate(true_labels, axis=0)
    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen kappa: {kappa:.4f}\n")
    return kappa, accuracy

if __name__ == '__main__':
    # Use GPU device is possible
    device = select_device()
    print('Device:', device)

    # preprocessing_pipeline = None
    # preprocessing_pipeline =[ben_graham_preprocessing]
    # preprocessing_pipeline =[circle_cropping]
    # preprocessing_pipeline = [clahe_preprocessing]
    # preprocessing_pipeline =[gaussian_blur]
    # preprocessing_pipeline =[sharpen_image]
    # preprocessing_pipeline = [circle_cropping, sharpen_image, gaussian_blur]
    preprocessing_pipeline = [circle_cropping, gaussian_blur]


    print('\nGenerating test data to verify on Kraggle\n') # No need to save for ensemble models since they are performing worse than individual models

    test_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/test.csv', '../DeepDRiD/test/', transform_test, preprocessing_pipeline, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    saved_models = load_saved_models(num_classes, device)

    model_to_test = saved_models[0] # densenet121

    evaluate_model(model_to_test, test_loader, device, criterion=None, test_only=True, prediction_path='outputs/test_predictions.csv')