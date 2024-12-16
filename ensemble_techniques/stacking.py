import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier


def get_predictions(model, loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_predictions.append(preds)

    return np.concatenate(all_predictions, axis=0)

def stacking_ensemble(saved_models, train_loader, val_loader, device):
    pred_model1 = get_predictions(saved_models[0], train_loader, device)
    pred_model2 = get_predictions(saved_models[1], train_loader, device)
    pred_model3 = get_predictions(saved_models[2], train_loader, device)

    stacked_predictions = np.hstack([pred_model1, pred_model2, pred_model3])

    true_labels = []
    for _, labels in train_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    stacking_ensemble = RandomForestClassifier()
    stacking_ensemble.fit(stacked_predictions, true_labels)

    pred_model1 = get_predictions(saved_models[0], val_loader, device)
    pred_model2 = get_predictions(saved_models[1], val_loader, device)
    pred_model3 = get_predictions(saved_models[2], val_loader, device)

    stacked_predictions = np.hstack([pred_model1, pred_model2, pred_model3])
    final_predictions = stacking_ensemble.predict(stacked_predictions)
    return final_predictions
