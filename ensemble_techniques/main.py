import torch
from torch.utils.data import DataLoader
from torchvision import models

from ensemble_techniques.bagging import bagging_ensemble, validate_bagging_ensemble
from ensemble_techniques.boosting import boosting_ensemble, validate_boosting_ensemble
from ensemble_techniques.max_voting import max_voting_ensemble, validate_max_voting_ensemble
from ensemble_techniques.stacking import stacking_ensemble, validate_stacking_ensemble
from ensemble_techniques.weighted_average import weighted_voting_ensemble, validate_weighted_voting_ensemble
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


if __name__ == '__main__':
    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ensemble_technique = 'bagging'
    assert ensemble_technique in ['stacking', 'boosting', 'max_voting', 'weighted_average', 'bagging']
    print('Ensemble technique:', ensemble_technique)

    match ensemble_technique:
        case 'stacking':
            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_train)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            stacking_ensemble = stacking_ensemble(saved_models, train_loader, device)
            validate_stacking_ensemble(stacking_ensemble, saved_models, val_loader, device)

        case 'boosting':
            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            final_predictions = boosting_ensemble(saved_models, train_loader, val_loader, device, num_classes)
            validate_boosting_ensemble(final_predictions, val_loader)

        case 'max_voting':
            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            final_predictions = max_voting_ensemble(saved_models, val_loader, device)
            validate_max_voting_ensemble(final_predictions, val_loader)

        case 'weighted_average': # this needs to be changed, currently using weighted voting not weighted average
            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            final_predictions = weighted_voting_ensemble(saved_models, val_loader, device, num_classes)
            validate_weighted_voting_ensemble(final_predictions, val_loader)

        case 'bagging':
            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            final_predictions = bagging_ensemble(saved_models, train_loader, val_loader, device)
            validate_bagging_ensemble(final_predictions, val_loader)
