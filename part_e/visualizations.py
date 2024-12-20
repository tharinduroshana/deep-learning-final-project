from matplotlib import pyplot as plt


def visualize_losses_and_accuracies(train_losses, val_losses, train_accuracies, val_accuracies, output_path='./loss_and_accuracy.png'):
    epochs = range(1, len(train_losses) + 1)

    # visualize loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # visualize accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    plt.tight_layout()
    plt.show()


def visualize_kappa(pre_processing_methods, model_names, all_kappas, output_path='./kappa_values_with_preprocessing.png'):
    assert len(model_names) == len(all_kappas)
    x_values = range(1, len(pre_processing_methods) + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(x_values, all_kappas[i], label=model_name)
        plt.xticks(x_values, pre_processing_methods)
    plt.title('Kappa values with Image Preprocessing')
    plt.xlabel('Image Preprocessing Techniques')
    plt.ylabel('Cohen\'s kappa')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.tight_layout()
    plt.show()
