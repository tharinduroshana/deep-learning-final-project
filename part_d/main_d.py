import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from part_d.ensemble_techniques.bagging import bagging_ensemble
from part_d.ensemble_techniques.boosting import boosting_ensemble
from part_d.ensemble_techniques.max_voting import max_voting_ensemble
from part_d.model import SecondStageModel
from part_d.ensemble_techniques.stacking import stacking_ensemble
from part_d.ensemble_techniques.weighted_average import weighted_average_ensemble
from part_d.dataset import PatientLevelRetinopathyDataset
from part_d.image_pre_processing.image_pre_processing import ben_graham_preprocessing, circle_cropping, \
    clahe_preprocessing, gaussian_blur, sharpen_image
from part_e.visualizations import visualize_kappa
from shared.shared import transform_test, transform_train, select_device, batch_size, num_classes, evaluate_model


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

    saved_models = load_saved_models(num_classes, device)

    preprocessing_pipelines = {
        "none": None,
        "ben_graham": [ben_graham_preprocessing],
        "circle_cropping": [circle_cropping],
        "clahe": [clahe_preprocessing],
        "gaussian_blur": [gaussian_blur],
        "sharpening": [sharpen_image],
    }

    # check_model_type = 'individual'
    check_model_type = 'ensemble'

    if check_model_type == 'individual':
        pre_processing_methods = []
        all_kappas = []
        model_names = []
        densenet121_kappas = []
        efficientnet_kappas = []
        resnet34_kappas = []
        for name, preprocess_funcs in preprocessing_pipelines.items():
            print(f"\nEvaluating Ensemble Techniques with Preprocessing: {name}\n")
            pre_processing_methods.append(name)

            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train, preprocess_funcs)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test, preprocess_funcs)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)
            densenet121 = saved_models[0]
            efficientnet_b0 = saved_models[1]
            resnet34 = saved_models[2]

            # Evaluate each individual model
            print('Individual model: densenet121')
            criterion = nn.CrossEntropyLoss()
            val_metrics = evaluate_model(densenet121, val_loader, device, criterion, test_only=False)
            val_kappa, val_accuracy, val_precision, val_recall, val_epoch_loss = val_metrics[:5]
            print(f"Accuracy: {val_accuracy:.4f}, Cohen's kappa: {val_kappa:.4f}")
            densenet121_kappas.append(val_kappa)

            print('Individual model: efficientnet_b0')
            criterion = nn.CrossEntropyLoss()
            val_metrics = evaluate_model(efficientnet_b0, val_loader, device, criterion, test_only=False)
            val_kappa, val_accuracy, val_precision, val_recall, val_epoch_loss = val_metrics[:5]
            print(f"Accuracy: {val_accuracy:.4f}, Cohen's kappa: {val_kappa:.4f}")
            efficientnet_kappas.append(val_kappa)

            print('Individual model: resnet34')
            criterion = nn.CrossEntropyLoss()
            val_metrics = evaluate_model(resnet34, val_loader, device, criterion, test_only=False)
            val_kappa, val_accuracy, val_precision, val_recall, val_epoch_loss = val_metrics[:5]
            print(f"Accuracy: {val_accuracy:.4f}, Cohen's kappa: {val_kappa:.4f}")
            resnet34_kappas.append(val_kappa)

        model_names.append('densenet121')
        all_kappas.append(densenet121_kappas)
        model_names.append('efficientnet_b0')
        all_kappas.append(efficientnet_kappas)
        model_names.append('resnet34')
        all_kappas.append(resnet34_kappas)

        color_options = ['b-o', 'g-o', 'r-o']
        visualize_kappa(pre_processing_methods, model_names, all_kappas, color_options, output_path='outputs/image_preprocessing_with_base_models.png')

    if check_model_type == 'ensemble':
        pre_processing_methods = []
        all_kappas = []
        model_names = []
        stacking_kappas = []
        boosting_kappas = []
        max_voting_kappas = []
        weighted_average_kappas = []
        bagging_kappas = []
        for name, preprocess_funcs in preprocessing_pipelines.items():
            print(f"\nEvaluating Ensemble Techniques with Preprocessing: {name}\n")
            pre_processing_methods.append(name)

            train_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/train.csv', '../DeepDRiD/train/', transform_train, preprocess_funcs)
            val_dataset = PatientLevelRetinopathyDataset('../DeepDRiD/val.csv', '../DeepDRiD/val/', transform_test, preprocess_funcs)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            saved_models = load_saved_models(num_classes, device)

            # Evaluate each ensemble technique
            print('Ensemble technique: stacking')
            final_predictions = stacking_ensemble(saved_models, train_loader, val_loader, device)
            val_metrics = validate_ensemble(final_predictions, val_loader)
            val_kappa, val_accuracy = val_metrics[:2]
            stacking_kappas.append(val_kappa)

            print('Ensemble technique: boosting')
            final_predictions = boosting_ensemble(saved_models, train_loader, val_loader, device, num_classes)
            val_metrics = validate_ensemble(final_predictions, val_loader)
            val_kappa, val_accuracy = val_metrics[:2]
            boosting_kappas.append(val_kappa)

            print('Ensemble technique: max voting')
            final_predictions = max_voting_ensemble(saved_models, val_loader, device)
            val_metrics = validate_ensemble(final_predictions, val_loader)
            val_kappa, val_accuracy = val_metrics[:2]
            max_voting_kappas.append(val_kappa)

            print('Ensemble technique: weighted average')
            final_predictions = weighted_average_ensemble(saved_models, val_loader, device, num_classes)
            val_metrics = validate_ensemble(final_predictions, val_loader)
            val_kappa, val_accuracy = val_metrics[:2]
            weighted_average_kappas.append(val_kappa)

            print('Ensemble technique: bagging')
            final_predictions = bagging_ensemble(saved_models, train_loader, val_loader, device)
            val_metrics = validate_ensemble(final_predictions, val_loader)
            val_kappa, val_accuracy = val_metrics[:2]
            bagging_kappas.append(val_kappa)

        model_names.append('stacking')
        all_kappas.append(stacking_kappas)
        model_names.append('boosting')
        all_kappas.append(boosting_kappas)
        model_names.append('max voting')
        all_kappas.append(max_voting_kappas)
        model_names.append('weighted average')
        all_kappas.append(weighted_average_kappas)
        model_names.append('bagging')
        all_kappas.append(bagging_kappas)

        color_options = ['b-o', 'g-o', 'r-o', 'm-o', 'k-o']
        visualize_kappa(pre_processing_methods, model_names, all_kappas, color_options, output_path='outputs/image_preprocessing_with_ensemble_models.png')
