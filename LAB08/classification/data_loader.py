"""โหลดรูปจาก Intel Image Classification ZIP"""
from pathlib import Path
import zipfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset"
ZIP_FILE = PROJECT_DIR / "archive (7).zip"


def prepare_dataset():
    """แตก ZIP เฉพาะภาพ train/test ครั้งแรก และข้ามรูปที่เสีย"""
    train_dir = DATA_DIR / "seg_train" / "seg_train"
    test_dir = DATA_DIR / "seg_test" / "seg_test"
    if not (train_dir.exists() and test_dir.exists()):
        if not ZIP_FILE.exists():
            raise FileNotFoundError(
                "ไม่พบ archive (7).zip กรุณาวางไฟล์ไว้ใน ML-08-DCNN/"
            )
        print("กำลังแตกไฟล์ Dataset ครั้งแรก (อาจใช้เวลาสักครู่)...")
        DATA_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(ZIP_FILE) as zf:
            for member in zf.infolist():
                if member.filename.startswith(("seg_train/", "seg_test/")):
                    dest = (DATA_DIR / member.filename).resolve()
                    if not dest.is_relative_to(DATA_DIR.resolve()):
                        raise ValueError("Unsafe path in ZIP")
                    zf.extract(member, DATA_DIR)
    # ตรวจสอบไฟล์รูปเสียแบบง่าย ๆ (ไม่ต้องก๊อปปี้ภาพไปอีกชุด)
    from PIL import Image
    for folder in (train_dir, test_dir):
        for file in folder.rglob("*.jpg"):
            try:
                with Image.open(file) as img:
                    img.verify()
            except Exception:
                print("ข้ามรูปที่เปิดไม่ได้:", file.name)
                file.unlink()
    return train_dir, test_dir
