import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate_model_accuracy(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def weighted_average_ensemble(models, val_loader, device, num_classes):
    model_accuracies = []
    for model in models:
        accuracy = evaluate_model_accuracy(model, val_loader, device)
        model_accuracies.append(accuracy)

    all_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = [image.to(device) for image in images]

            model_preds = []
            for model in models:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                model_preds.append(preds)

            model_preds = np.array(model_preds).T
            all_predictions.append(model_preds)

    final_predictions = []
    for model_preds in np.concatenate(all_predictions, axis=0):
        weighted_votes = np.zeros(num_classes)

        for i, pred in enumerate(model_preds):
            weighted_votes[pred] += model_accuracies[i]

        final_prediction = np.argmax(weighted_votes)
        final_predictions.append(final_prediction)

    return np.array(final_predictions)
