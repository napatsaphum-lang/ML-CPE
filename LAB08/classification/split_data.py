"""แบ่ง train / validation และโหลด test แยกต่างหาก"""
import tensorflow as tf
from .preprocessing import IMAGE_SIZE, BATCH_SIZE, SEED


def load_datasets(train_dir, test_dir):
    common = dict(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED)
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", **common
    )
    val = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", **common
    )
    classes = train.class_names
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_names=classes,
    )
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), val.prefetch(autotune), test.prefetch(autotune), classes
