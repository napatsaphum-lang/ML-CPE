"""เปิดโมเดลที่ฝึกไว้แล้ว เพื่อดูภาพตัวอย่างทำนาย"""
import json
from pathlib import Path
import tensorflow as tf
from .data_loader import prepare_dataset
from .split_data import load_datasets
from .evaluate import save_evaluation


def main():
    folder = Path(__file__).resolve().parent / "outputs"
    model_file = folder / "best_model.keras"
    if not model_file.exists():
        print("กรุณารัน python -m classification.main ก่อน")
        return
    train_dir, test_dir = prepare_dataset()
    _, _, test_ds, classes = load_datasets(train_dir, test_dir)
    model = tf.keras.models.load_model(model_file)
    save_evaluation(model, test_ds, classes, folder)
    print("สร้างภาพ predictions.png และ confusion_matrix.png ใหม่แล้ว")


if __name__ == "__main__":
    main()
