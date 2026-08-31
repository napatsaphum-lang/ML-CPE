import os
import numpy as np

from tensorflow.keras.models import load_model

from data_loader import load_data

from preprocessing import (
    prepare_data,
    standardize_data
)

from split_data import split_dataset


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "best_model.keras"
)


# -----------------------------
# Load Dataset
# -----------------------------

data = load_data(
    DATA_PATH
)


# -----------------------------
# Preprocessing
# -----------------------------

X, y, encoder = prepare_data(
    data
)


# -----------------------------
# Split Dataset
# -----------------------------

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
    test_size=0.2,
    val_size=0.1
)


# -----------------------------
# Standardization
# -----------------------------

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


# -----------------------------
# Load Best Model
# -----------------------------

model = load_model(
    MODEL_PATH
)


# -----------------------------
# Prediction
# -----------------------------

probability = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probability,
    axis=1
)


# -----------------------------
# Show Sample Predictions
# -----------------------------

print("\n====================================")
print("Sample Predictions")
print("====================================")


for i in range(10):

    actual = encoder.inverse_transform(
        [y_test[i]]
    )[0]

    predicted = encoder.inverse_transform(
        [predictions[i]]
    )[0]

    confidence = np.max(
        probability[i]
    )


    print(
        "\nSample",
        i + 1
    )

    print(
        "Actual    :",
        actual
    )

    print(
        "Predicted :",
        predicted
    )

    print(
        "Confidence:",
        round(
            confidence * 100,
            2
        ),
        "%"
    )

    print(
        "------------------------------------"
    )