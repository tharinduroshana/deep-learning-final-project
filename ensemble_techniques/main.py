import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision import models

from ensemble_techniques.bagging import bagging_ensemble
from ensemble_techniques.boosting import boosting_ensemble
from ensemble_techniques.max_voting import max_voting_ensemble
from ensemble_techniques.stacking import stacking_ensemble
from ensemble_techniques.weighted_average import weighted_average_ensemble
from patient_level_retinopthy_dataset import PatientLevelRetinopathyDataset
from second_stage_model import SecondStageModel
from template_code import transform_test, transform_train

batch_size = 24
num_classes = 5

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
    state_dict = torch.load('../saved_models/densenet121.pth', map_location=device, weights_only=True)
    model_densenet121.load_state_dict(state_dict, strict=True)

    model_efficientnet_b0 = SecondStageModel(saved_models["efficientnet_b0"], in_features["efficientnet_b0"], num_classes=num_classes).to(device)
    state_dict = torch.load('../saved_models/efficientnet_b0.pth', map_location=device, weights_only=True)
    model_efficientnet_b0.load_state_dict(state_dict, strict=True)

    model_resnet34 = SecondStageModel(saved_models["resnet34"], in_features["resnet34"], num_classes=num_classes).to(device)
    state_dict = torch.load('../saved_models/resnet34.pth', map_location=device, weights_only=True)
    model_resnet34.load_state_dict(state_dict, strict=True)

    return [model_densenet121, model_efficientnet_b0, model_resnet34]

def validate_ensemble(final_predictions, val_loader):
    true_labels = [labels.numpy() for _, labels in val_loader]
    true_labels = np.concatenate(true_labels, axis=0)
    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen kappa: {kappa:.4f}\n")

if __name__ == '__main__':
    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Evaluating Ensemble Techniques:\n')

    train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train)
    val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    saved_models = load_saved_models(num_classes, device)

    print('Ensemble technique: stacking')
    final_predictions = stacking_ensemble(saved_models, train_loader, val_loader, device)
    validate_ensemble(final_predictions, val_loader)

    print('Ensemble technique: boosting')
    final_predictions = boosting_ensemble(saved_models, train_loader, val_loader, device, num_classes)
    validate_ensemble(final_predictions, val_loader)

    print('Ensemble technique: max voting')
    final_predictions = max_voting_ensemble(saved_models, val_loader, device)
    validate_ensemble(final_predictions, val_loader)

    print('Ensemble technique: weighted average')
    final_predictions = weighted_average_ensemble(saved_models, val_loader, device, num_classes)
    validate_ensemble(final_predictions, val_loader)

    print('Ensemble technique: bagging')
    final_predictions = bagging_ensemble(saved_models, train_loader, val_loader, device)
    validate_ensemble(final_predictions, val_loader)
