import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import models

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
    state_dict = torch.load('./saved_models/densenet121.pth', map_location=device, weights_only=True)
    model_densenet121.load_state_dict(state_dict, strict=True)

    model_efficientnet_b0 = SecondStageModel(saved_models["efficientnet_b0"], in_features["efficientnet_b0"], num_classes=num_classes).to(device)
    state_dict = torch.load('./saved_models/efficientnet_b0.pth', map_location=device, weights_only=True)
    model_efficientnet_b0.load_state_dict(state_dict, strict=True)

    model_resnet34 = SecondStageModel(saved_models["resnet34"], in_features["resnet34"], num_classes=num_classes).to(device)
    state_dict = torch.load('./saved_models/resnet34.pth', map_location=device, weights_only=True)
    model_resnet34.load_state_dict(state_dict, strict=True)

    return [model_densenet121, model_efficientnet_b0, model_resnet34]


def get_predictions(model, loader, device, test_only=False):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            if test_only:
                images = batch
            else:
                images, _ = batch

            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def get_trained_stack_ensemble(saved_models, train_loader, device):
    pred_model1 = get_predictions(saved_models[0], train_loader, device)
    pred_model2 = get_predictions(saved_models[1], train_loader, device)
    pred_model3 = get_predictions(saved_models[2], train_loader, device)

    stacked_predictions = np.hstack([pred_model1, pred_model2, pred_model3])

    true_labels = []
    for _, labels in train_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    stack_ensemble = RandomForestClassifier()
    stack_ensemble.fit(stacked_predictions, true_labels)
    return stack_ensemble


def test_stack_ensemble(trained_ensemble, saved_models, val_loader, device, test_only=False):
    pred_model1 = get_predictions(saved_models[0], val_loader, device, test_only)
    pred_model2 = get_predictions(saved_models[1], val_loader, device, test_only)
    pred_model3 = get_predictions(saved_models[2], val_loader, device, test_only)

    stacked_predictions = np.hstack([pred_model1, pred_model2, pred_model3])
    final_predictions = trained_ensemble.predict(stacked_predictions)

    if not test_only:
        test_labels = []
        for batch in val_loader:
            _, labels = batch  # Test set is labeled
            test_labels.extend(labels.numpy())
        test_labels = np.array(test_labels)

        accuracy = accuracy_score(test_labels, final_predictions)
        kappa = cohen_kappa_score(test_labels, final_predictions)

        print(f"Stacking Ensemble Test Accuracy: {accuracy:.4f}")
        print(f"Stacking Ensemble Test Cohen Kappa: {kappa:.4f}")
    else:
        print("Test set is unlabeled. Predictions are generated.")
        print(final_predictions)
        np.save("test_predictions.npy", final_predictions)
        # df = pd.DataFrame({
        #     'ID': all_image_ids,
        #     'TARGET': final_predictions
        # })
        # df.to_csv(prediction_path, index=False)

if __name__ == '__main__':
    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Save test predictions
    val_dataset = PatientLevelRetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
    test_dataset = PatientLevelRetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, test=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    saved_models = load_saved_models(num_classes, device)
    trained_stack_ensemble = get_trained_stack_ensemble(saved_models, val_loader, device)
    test_stack_ensemble(trained_stack_ensemble, saved_models, test_loader, device, test_only=True)

    # Using training dataset to train the ensemble model  can cause over fitting as it is already used at individual models level
    # Test dataset is not labeled, therefore, cannot measure any metrics
    # So we have two ways:
    # 1. Use validation set with data augmentation
    # 2. K-Fold with validation data

    # Data augmentation
    # train_dataset = PatientLevelRetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_train)
    # val_dataset = PatientLevelRetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    # saved_models = load_saved_models(num_classes, device)
    # trained_stack_ensemble = get_trained_stack_ensemble(saved_models, train_loader, device)
    # test_stack_ensemble(trained_stack_ensemble, saved_models, val_loader, device, test_only=False)

    # K-Fold
    # k_folds = 5
    # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    #
    # saved_models = load_saved_models(num_classes, device)
    # val_dataset = PatientLevelRetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
    # for fold, (train_indices, val_indices) in enumerate(kf.split(val_dataset)):
    #     print(f"Fold {fold + 1}/{k_folds}")
    #
    #     # Create train and validation subsets
    #     train_subset = Subset(val_dataset, train_indices)
    #     val_subset = Subset(val_dataset, val_indices)
    #
    #     # Create DataLoaders
    #     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    #
    #     # Train models or ensemble using this fold's training data
    #     trained_stack_ensemble = get_trained_stack_ensemble(saved_models, train_loader, device)
    #
    #     # Evaluate using this fold's validation data
    #     test_stack_ensemble(trained_stack_ensemble, saved_models, val_loader, device)
