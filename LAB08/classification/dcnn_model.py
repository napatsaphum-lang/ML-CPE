"""สร้าง DCNN แบบเข้าใจง่าย ปรับ Conv Layers และ Dense Neurons ได้"""
import tensorflow as tf
from .preprocessing import IMAGE_SIZE


def build_model(num_conv=3, neurons=128, num_classes=6):
    model = tf.keras.Sequential(name=f"DCNN_{num_conv}conv_{neurons}dense")
    model.add(tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)))
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    # เพิ่มจำนวน filters ทีละชั้น โดยไม่ให้เกิน 128
    for i in range(num_conv):
        filters = min(32 * (2 ** i), 128)
        model.add(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(neurons, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
