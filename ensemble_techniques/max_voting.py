import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score


def max_voting_ensemble(models, val_loader, device):
    all_predictions = []  # To store the predictions from each model

    # Collect predictions from each model
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = [image.to(device) for image in images]

            model_preds = []  # To store the predictions of each model for this batch
            for model in models:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get class predictions
                model_preds.append(preds)

            # Convert list of predictions into an array (one row per sample)
            model_preds = np.array(model_preds).T  # Shape: (batch_size, n_models)

            # Apply majority voting (mode across predictions)
            final_preds = np.array([np.bincount(sample_preds).argmax() for sample_preds in model_preds])

            all_predictions.extend(final_preds)

    # Convert final predictions list to a numpy array
    return np.array(all_predictions)


def validate_max_voting_ensemble(final_predictions, val_loader):
    true_labels = [labels.numpy() for _, labels in val_loader]
    true_labels = np.concatenate(true_labels, axis=0)
    accuracy = accuracy_score(true_labels, final_predictions)
    kappa = cohen_kappa_score(true_labels, final_predictions)

    print(f"Max Voting Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Max Voting Ensemble Validation Cohen Kappa: {kappa:.4f}")