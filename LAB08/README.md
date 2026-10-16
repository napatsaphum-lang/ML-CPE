# ML LAB 08 — DCNN (Intel Image Classification)

โปรเจกต์ตัวอย่างแบบแยกไฟล์ตามหน้าที่ ใช้ Python + TensorFlow สำหรับนักศึกษาทดลอง DCNN ใน VS Code

## โครงสร้าง

```text
ML-08-DCNN/
├── archive (7).zip                # นำ Dataset จาก Kaggle มาวางเอง
├── dataset/                      # ระบบจะแตกไฟล์ seg_train และ seg_test อัตโนมัติ
├── classification/
│   ├── __init__.py
│   ├── main.py                   # โปรแกรมหลัก: ฝึกและเปรียบเทียบ 6 รอบ
│   ├── data_loader.py            # อ่าน ZIP ตรวจภาพเสีย
│   ├── preprocessing.py          # ขนาดภาพและ batch size
│   ├── split_data.py             # แบ่ง Train / Validation / Test
│   ├── dcnn_model.py             # สร้าง DCNN และกำหนด Layers
│   ├── evaluate.py               # สร้างกราฟ, confusion matrix, CSV
│   ├── test_model.py             # ใช้โมเดลที่ฝึกแล้วทำนายภาพ
│   └── outputs/                 # ไฟล์ผลลัพธ์ (สร้างหลังรัน)
│       ├── classes.json
│       ├── best_model.keras
│       ├── comparison.csv
│       ├── comparison.png
│       ├── history_A.png ... history_F.png
│       ├── confusion_matrix.png
│       ├── classification_report.txt
│       └── predictions.png
└── requirements.txt
```

## เริ่มใช้ใน VS Code (Windows, แนะนำ Python 3.11)
1. แตก ZIP โปรเจกต์แล้วเปิดโฟลเดอร์ `ML-08-DCNN` ใน VS Code
2. คัดลอกไฟล์ Dataset `archive (7).zip` ไปไว้ในโฟลเดอร์ `ML-08-DCNN`
3. เปิด Terminal ที่โฟลเดอร์โปรเจกต์:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m classification.main --quick
```

4. ถ้าทดสอบผ่าน ให้รันจริง:

```powershell
python -m classification.main
```

5. ทดสอบโมเดลที่บันทึกไว้ (ไม่ฝึกใหม่):

```powershell
python -m classification.test_model
```

> หาก PowerShell ไม่ยอม Activate ให้ใช้ `.venv\Scripts\python.exe -m classification.main --quick` โดยไม่ต้อง Activate

## ตารางการทดลอง

| Model | Conv Layers | Dense Neurons | Epochs |
|---|---:|---:|---:|
| A | 2 | 128 | 10 |
| B | 3 | 128 | 10 |
| C | 4 | 128 | 10 |
| D | 3 | 128 | 20 |
| E | 3 | 128 | 30 |
| F | 3 | 256 | 10 |

โจทย์ LAB ต้องการ Accuracy แยกแต่ละรูปแบบ, กราฟ Train/Validation Accuracy-Loss และตัวอย่าง Predictions

**หมายเหตุ:** โค้ดนี้เป็นงานฝึกทำแบบง่าย ไม่ได้ใช้ Transfer Learning หรือ VGG เพราะ LAB ไม่ได้บังคับ โมเดลที่บันทึกใช้ Validation Accuracy เป็นหลักในการเลือก แต่ Test Accuracy ใช้รายงานผล หากเครื่องช้าอาจลด IMAGE_SIZE หรือทดลองทีละโมเดลก่อน
