import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.utils import resample


def bagging_ensemble(models, train_loader, val_loader, device):
    """
    Bagging Ensemble: Trains n models on bootstrap samples of the training data and uses majority voting for predictions.

    Args:
        models (list): List of pre-trained models.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device (CPU or GPU).
        num_classes (int): Number of output classes.
        n_models (int): Number of models in the ensemble (default is 5).

    Returns:
        np.ndarray: Final predictions for the validation set.
    """
    all_predictions = []  # To store predictions from each model

    for model in models:
        # Create bootstrap sample for the training data
        bootstrap_samples = resample(train_loader.dataset, replace=True, n_samples=len(train_loader.dataset))
        bootstrap_loader = torch.utils.data.DataLoader(bootstrap_samples, batch_size=train_loader.batch_size, shuffle=True)

        model = model.to(device)
        model.train()  # Set the model to training mode

        # Train the model on the bootstrap sample
        for batch in bootstrap_loader:
            images, labels = batch
            images = [image.to(device) for image in images]
            labels = labels.to(device)

            # Forward pass and compute loss (assuming you have a loss function and optimizer)
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            # Backpropagate and optimize the model (add optimizer step here)

        # After training, make predictions on the validation set
        model.eval()  # Set the model to evaluation mode
        batch_preds = []

        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = [image.to(device) for image in images]

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get class predictions
                batch_preds.append(preds)

        all_predictions.append(np.concatenate(batch_preds, axis=0))  # Append model predictions

    # Now apply majority voting across all models
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
    final_predictions = []

    for sample_preds in all_predictions.T:  # Iterate over all samples
        final_pred = np.bincount(sample_preds).argmax()  # Majority vote
        final_predictions.append(final_pred)

    return np.array(final_predictions)


def validate_bagging_ensemble(final_predictions, val_loader):
    """
    Validate the Bagging ensemble on the validation dataset.

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

    print(f"Bagging Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Bagging Ensemble Validation Cohen Kappa: {kappa:.4f}")
