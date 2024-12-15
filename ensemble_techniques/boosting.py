import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import torch


def boosting_ensemble(models, train_loader, val_loader, device, num_classes):
    n_samples = len(train_loader.dataset)
    sample_weights = np.ones(n_samples) / n_samples  # Initialize sample weights

    model_weights = []  # To store weights for each model
    final_predictions = np.zeros((len(val_loader.dataset), num_classes))  # Ensemble predictions

    for i, model in enumerate(models):
        print(f"Boosting Round {i + 1}")

        # Collect predictions for the model
        train_preds = []
        train_labels = []
        with torch.no_grad():
            for batch in train_loader:
                images, labels = batch
                images = [image.to(device) for image in images]
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                train_preds.append(preds)
                train_labels.extend(labels.numpy())

        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.array(train_labels)

        # Compute weighted error for this model
        pred_labels = np.argmax(train_preds, axis=1)
        incorrect = (pred_labels != train_labels).astype(int)
        weighted_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

        # Compute model weight (alpha)
        if weighted_error == 0:  # Perfect model
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
        model_weights.append(alpha)

        # Update sample weights
        sample_weights *= np.exp(alpha * incorrect)
        sample_weights /= np.sum(sample_weights)  # Normalize weights

        # Update final predictions
        with torch.no_grad():
            start_idx = 0
            for batch in val_loader:
                images, labels = batch
                batch_size = labels.size(0)
                images = [image.to(device) for image in images]
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1).cpu().numpy()

            final_predictions[start_idx:start_idx + batch_size] += alpha * preds
            start_idx += batch_size

    # Return the final predictions
    final_predictions = np.argmax(final_predictions, axis=1)
    return final_predictions

def validate_boosting_ensemble(final_predictions, val_loader):
    true_labels = [labels.numpy() for _, labels in val_loader]
    true_labels = np.concatenate(true_labels, axis=0)
    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Boosting Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Boosting Ensemble Validation Cohen Kappa: {kappa:.4f}")