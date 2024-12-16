import numpy as np
import torch
from sklearn.utils import resample


def bagging_ensemble(models, train_loader, val_loader, device):
    all_predictions = []  # To store predictions from each model
    for model in models:
        bootstrap_samples = resample(train_loader.dataset, replace=True, n_samples=len(train_loader.dataset))
        bootstrap_loader = torch.utils.data.DataLoader(bootstrap_samples, batch_size=train_loader.batch_size, shuffle=True)

        model = model.to(device)
        model.train()

        for batch in bootstrap_loader:
            images, labels = batch
            images = [image.to(device) for image in images]
            labels = labels.to(device)

            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        model.eval()
        batch_preds = []
        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = [image.to(device) for image in images]

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get class predictions
                batch_preds.append(preds)

        all_predictions.append(np.concatenate(batch_preds, axis=0))  # Append model predictions

    all_predictions = np.array(all_predictions)
    final_predictions = []

    for sample_preds in all_predictions.T:
        final_pred = np.bincount(sample_preds).argmax()
        final_predictions.append(final_pred)

    return np.array(final_predictions)
