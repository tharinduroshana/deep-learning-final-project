import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)

def stacking_ensemble(saved_models, train_loader, device):
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


def validate_stacking_ensemble(trained_ensemble, saved_models, val_loader, device):
    pred_model1 = get_predictions(saved_models[0], val_loader, device)
    pred_model2 = get_predictions(saved_models[1], val_loader, device)
    pred_model3 = get_predictions(saved_models[2], val_loader, device)

    stacked_predictions = np.hstack([pred_model1, pred_model2, pred_model3])
    final_predictions = trained_ensemble.predict(stacked_predictions)

    true_labels = [labels.numpy() for _, labels in val_loader]
    true_labels = np.concatenate(true_labels, axis=0)
    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Stacking Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Stacking Ensemble Validation Cohen Kappa: {kappa:.4f}")
