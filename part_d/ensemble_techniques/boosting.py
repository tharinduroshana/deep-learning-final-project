import numpy as np
import torch


def boosting_ensemble(models, train_loader, val_loader, device, num_classes):
    n_samples = len(train_loader.dataset)
    sample_weights = np.ones(n_samples) / n_samples

    model_weights = []
    final_predictions = np.zeros((len(val_loader.dataset), num_classes))
    for model in models:
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

        pred_labels = np.argmax(train_preds, axis=1)
        incorrect = (pred_labels != train_labels).astype(int)
        weighted_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

        if weighted_error == 0:  # Perfect model
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
        model_weights.append(alpha)

        sample_weights *= np.exp(alpha * incorrect)
        sample_weights /= np.sum(sample_weights)  # Normalize weights

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

    final_predictions = np.argmax(final_predictions, axis=1)
    return final_predictions
