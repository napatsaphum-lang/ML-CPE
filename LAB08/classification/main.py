"""LAB 08: Intel Image Classification ด้วย DCNN (รันจากโฟลเดอร์โปรเจกต์)"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from .data_loader import prepare_dataset
from .split_data import load_datasets
from .dcnn_model import build_model
from .evaluate import plot_history, save_evaluation, save_comparison

OUT = Path(__file__).resolve().parent / "outputs"


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    train_dir, test_dir = prepare_dataset()
    train_ds, val_ds, test_ds, classes = load_datasets(train_dir, test_dir)
    (OUT / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # เปลี่ยนทีละตัวแปรเพื่อเปรียบเทียบผลตามโจทย์ LAB 08
    experiments = [
        ("A", 2, 128, 10),
        ("B", 3, 128, 10),
        ("C", 4, 128, 10),
        ("D", 3, 128, 20),
        ("E", 3, 128, 30),
        ("F", 3, 256, 10),
    ]
    if quick:
        experiments = [("QUICK", 2, 64, 1)]
    results = []
    best_accuracy = -1
    for name, num_conv, neurons, epochs in experiments:
        print(f"\n===== Model {name}: Conv={num_conv}, Neurons={neurons}, Epochs={epochs} =====")
        tf.keras.backend.clear_session()
        model = build_model(num_conv, neurons, len(classes))
        model.summary()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        plot_history(hist, OUT, name)
        row = {
            "model": name, "conv_layers": num_conv, "neurons": neurons, "epochs": epochs,
            "train_accuracy": round(float(hist.history["accuracy"][-1]), 4),
            "val_accuracy": round(float(hist.history["val_accuracy"][-1]), 4),
            "test_accuracy": round(float(accuracy), 4), "test_loss": round(float(loss), 4),
        }
        results.append(row)
        print("Result:", row)
        # เก็บไฟล์ของโมเดลตาม Validation Accuracy ไม่ใช่ใช้ Test ในการเลือก
        val_best = max(hist.history["val_accuracy"])
        if val_best > best_accuracy:
            best_accuracy = val_best
            model.save(OUT / "best_model.keras")
    save_comparison(results, OUT)
    best_model = tf.keras.models.load_model(OUT / "best_model.keras")
    save_evaluation(best_model, test_ds, classes, OUT)
    print("\nเสร็จแล้ว! ดูกราฟและตารางได้ที่ classification/outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="ทดสอบ 1 epoch ก่อน")
    args = parser.parse_args()
    main(quick=args.quick)
