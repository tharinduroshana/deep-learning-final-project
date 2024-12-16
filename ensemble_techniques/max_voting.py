import numpy as np
import torch


def max_voting_ensemble(models, val_loader, device):
    final_predictions = []

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

            final_preds = np.array([np.bincount(sample_preds).argmax() for sample_preds in model_preds])
            final_predictions.extend(final_preds)

    return np.array(final_predictions)
