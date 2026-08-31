import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_size, hidden_layers, num_classes):

    model = Sequential()

    # Hidden Layer แรก
    model.add(
        Dense(
            hidden_layers[0],
            activation="relu",
            input_shape=(input_size,)
        )
    )

    # Hidden Layer เพิ่มเติม
    for neuron in hidden_layers[1:]:

        model.add(
            Dense(
                neuron,
                activation="relu"
            )
        )

    # Output Layer
    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers,
    num_classes,
    epochs=50,
    batch_size=8
):

    model = build_model(
        X_train.shape[1],
        hidden_layers,
        num_classes
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model, history


def predict_model(model, X_test):

    probability = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probability,
        axis=1
    )

    return predictions