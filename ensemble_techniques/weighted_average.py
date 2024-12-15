import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate_model_accuracy(model, val_loader, device):
    """
    Evaluate the accuracy of a model on the validation set.

    Args:
        model (nn.Module): Pre-trained model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device (CPU or GPU).

    Returns:
        float: Accuracy of the model on the validation set.
    """
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


def weighted_voting_ensemble(models, val_loader, device, num_classes):
    """
    Weighted Voting Ensemble using pre-trained models based on accuracy.

    Args:
        models (list): List of pre-trained models.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device (CPU or GPU).
        num_classes (int): Number of output classes.

    Returns:
        np.ndarray: Final predictions for the validation set.
    """
    # 1. Evaluate accuracy of each model
    model_accuracies = []
    for model in models:
        accuracy = evaluate_model_accuracy(model, val_loader, device)
        model_accuracies.append(accuracy)
        print(f"Model Accuracy: {accuracy:.4f}")

    # 2. Collect predictions from each model
    all_predictions = []  # List to store predictions of each model for each sample

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = [image.to(device) for image in images]

            model_preds = []  # Store predictions for this batch from all models
            for model in models:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                model_preds.append(preds)

            # Stack model predictions for this batch (shape: batch_size x n_models)
            model_preds = np.array(model_preds).T  # Shape: (batch_size, n_models)

            all_predictions.append(model_preds)

    # 3. Apply weighted voting
    final_predictions = []
    for model_preds in np.concatenate(all_predictions, axis=0):  # Iterate over all samples
        weighted_votes = np.zeros(num_classes)  # Initialize vote counts for each class

        # Weighted sum of votes for each model (weighted by accuracy)
        for i, pred in enumerate(model_preds):
            weighted_votes[pred] += model_accuracies[i]

        # Select the class with the highest weighted vote
        final_prediction = np.argmax(weighted_votes)
        final_predictions.append(final_prediction)

    return np.array(final_predictions)


def validate_weighted_voting_ensemble(final_predictions, val_loader):
    """
    Validate the weighted voting ensemble on the validation dataset.

    Args:
        final_predictions (np.ndarray): Ensemble's final predictions.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        None: Prints accuracy and Cohen Kappa score.
    """
    true_labels = []
    for _, labels in val_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Weighted Voting Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Weighted Voting Ensemble Validation Cohen Kappa: {kappa:.4f}")
