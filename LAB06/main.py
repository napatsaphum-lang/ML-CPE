import os

from data_loader import load_data

from preprocessing import (
    prepare_data,
    standardize_data
)

from split_data import split_dataset

from nn_model import (
    train_model,
    predict_model
)

from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_history,
    plot_comparison
)


# -----------------------------
# Path
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Iris.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# -----------------------------
# Setting
# -----------------------------

TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 8


# -----------------------------
# Model Configuration
# -----------------------------

experiments = [

    {
        "name": "Model 1",
        "layers": [8],
        "epochs": 20
    },

    {
        "name": "Model 2",
        "layers": [8],
        "epochs": 50
    },

    {
        "name": "Model 3",
        "layers": [16, 8],
        "epochs": 50
    },

    {
        "name": "Model 4",
        "layers": [32, 16, 8],
        "epochs": 50
    }
]


def main():

    print("====================================")
    print("Neural Network - Iris Dataset")
    print("====================================")


    # สร้าง outputs folder
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # -------------------------
    # Step 1 Load Dataset
    # -------------------------

    print("\nStep 1: Load Dataset")

    data = load_data(
        DATA_PATH
    )


    # -------------------------
    # Step 2 Preprocessing
    # -------------------------

    print("\nStep 2: Preprocessing")

    X, y, encoder = prepare_data(
        data
    )


    # -------------------------
    # Step 3 Split Dataset
    # -------------------------

    print("\nStep 3: Split Dataset")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        X,
        y,
        TEST_SIZE,
        VAL_SIZE
    )


    print(
        "Training  :",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_val)
    )

    print(
        "Testing   :",
        len(X_test)
    )


    # -------------------------
    # Step 4 Standardization
    # -------------------------

    print("\nStep 4: Standardization")

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = standardize_data(
        X_train,
        X_val,
        X_test
    )


    # -------------------------
    # Step 5 Train Models
    # -------------------------

    print("\nStep 5: Train Models")

    model_names = []
    accuracies = []

    best_model = None
    best_history = None
    best_matrix = None

    best_accuracy = 0
    best_name = ""


    for experiment in experiments:

        print("\n====================================")

        print(
            experiment["name"]
        )

        print(
            "Hidden Layers:",
            experiment["layers"]
        )

        print(
            "Epochs:",
            experiment["epochs"]
        )

        print(
            "===================================="
        )


        # Train Model
        model, history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            experiment["layers"],
            len(encoder.classes_),
            experiment["epochs"],
            BATCH_SIZE
        )


        # Prediction
        predictions = predict_model(
            model,
            X_test
        )


        # Evaluation
        accuracy, matrix = evaluate_model(
            y_test,
            predictions
        )


        model_names.append(
            experiment["name"]
        )

        accuracies.append(
            accuracy
        )


        # เก็บ Model ที่ดีที่สุด
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_history = history
            best_matrix = matrix
            best_name = experiment["name"]


    # -------------------------
    # Step 6 Compare Models
    # -------------------------

    print("\n====================================")
    print("Model Comparison")
    print("====================================")


    for i in range(
        len(experiments)
    ):

        print(
            experiments[i]["name"],
            "| Layers:",
            experiments[i]["layers"],
            "| Epochs:",
            experiments[i]["epochs"],
            "| Accuracy:",
            round(
                accuracies[i] * 100,
                2
            ),
            "%"
        )


    print(
        "\nBest Model:",
        best_name
    )

    print(
        "Best Accuracy:",
        round(
            best_accuracy * 100,
            2
        ),
        "%"
    )


    # -------------------------
    # Step 7 Save Results
    # -------------------------

    print(
        "\nStep 7: Save Results"
    )


    # Model Comparison
    plot_comparison(
        model_names,
        accuracies,
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.png"
        )
    )


    # Confusion Matrix
    plot_confusion_matrix(
        best_matrix,
        encoder.classes_,
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )


    # Accuracy / Loss
    plot_history(
        best_history,

        os.path.join(
            OUTPUT_DIR,
            "training_accuracy.png"
        ),

        os.path.join(
            OUTPUT_DIR,
            "training_loss.png"
        )
    )


    # Save Best Model
    best_model.save(
        os.path.join(
            OUTPUT_DIR,
            "best_model.keras"
        )
    )


    print("\nFinished")
    print("Results saved in outputs folder")


if __name__ == "__main__":
    main()