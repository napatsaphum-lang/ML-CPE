import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


def evaluate_model(y_test, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    return accuracy, matrix


def plot_confusion_matrix(
    matrix,
    classes,
    save_path
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=20
    )

    plt.yticks(
        range(len(classes)),
        classes
    )

    for i in range(len(classes)):

        for j in range(len(classes)):

            plt.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        save_path
    )

    plt.close()


def plot_history(
    history,
    accuracy_path,
    loss_path
):

    # -------------------------
    # Accuracy Graph
    # -------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Training"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        accuracy_path
    )

    plt.close()


    # -------------------------
    # Loss Graph
    # -------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        loss_path
    )

    plt.close()


def plot_comparison(
    model_names,
    accuracies,
    save_path
):

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        model_names,
        accuracies
    )

    plt.title(
        "Neural Network Model Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        save_path
    )

    plt.close()