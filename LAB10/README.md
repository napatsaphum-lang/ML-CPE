# LAB10 – Object Detection Using YOLOv8

โปรเจกต์ LAB10 รายวิชา Machine Learning (04-624-201) ใช้ YOLOv8n ตรวจจับบอร์ดไมโครคอนโทรลเลอร์ 4 ประเภท และเปรียบเทียบผลเมื่อปรับ Epochs, Image Size และ Batch Size

## Dataset

Microcontroller Detection จาก Kaggle: https://www.kaggle.com/datasets/tannergi/microcontroller-detection

ประกอบด้วย 149 ภาพ และ Label แบบ Pascal VOC XML โปรแกรมจะแปลงเป็น YOLO TXT และแบ่ง Train/Val/Test อัตโนมัติ (Train 114, Val 28, Test 7 ภาพ) 4 Classes: ESP8266, Heltec ESP32 LoRa, Arduino Nano และ Raspberry Pi 3

## โครงสร้าง

```text
LAB10/
├── archive (8).zip            # วางไฟล์ Kaggle ที่นี่ (ไม่อัปโหลดขึ้น GitHub)
├── dataset/                  # YOLO dataset (สร้างอัตโนมัติ)
├── detection/
│   ├── data_loader.py        # แปลง XML และแบ่งข้อมูล
│   ├── train.py              # ฝึก YOLOv8n
│   ├── evaluate.py           # Precision, Recall, mAP, comparison
│   ├── main.py               # เรียกทุกขั้นตอน
│   ├── image_test.py         # ภาพนิ่ง
│   ├── video_test.py         # วิดีโอ
│   └── realtime_test.py      # กล้อง
├── outputs/                  # กราฟและผลการฝึก
├── requirements.txt
└── README.md
```

## วิธีรัน (VS Code – Windows)

เปิด Terminal ที่โฟลเดอร์ LAB10:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m detection.main --quick
```

ถ้าทดลองเร็วสำเร็จแล้ว รัน Training ทุกโมเดล:

```powershell
.\.venv\Scripts\python.exe -m detection.main
```

ทดลองบางโมเดล เช่น A และ B:

```powershell
.\.venv\Scripts\python.exe -m detection.main --models A B
```

ทดสอบภาพหลังฝึก Model A แบบเต็ม:

```powershell
.\.venv\Scripts\python.exe -m detection.image_test
```

ทดสอบวิดีโอ:

```powershell
.\.venv\Scripts\python.exe -m detection.video_test --source demo.mp4
```

ทดสอบ Webcam (กด Q หรือปิดหน้าต่างเพื่อหยุด):

```powershell
.\.venv\Scripts\python.exe -m detection.realtime_test
```

**หมายเหตุ:** คำสั่งที่อยู่ใน README ต้องรันจากโฟลเดอร์ LAB10 และใช้ `-m detection.main` ไม่ใช่กด Run Python File เพื่อหลีกเลี่ยงปัญหา relative imports

## การทดลอง

| Model | Epochs | Image size | Batch |
|---|---:|---:|---:|
| A | 10 | 416 | 8 |
| B | 20 | 416 | 8 |
| C | 30 | 416 | 8 |
| D | 20 | 640 | 8 |
| E | 20 | 416 | 16 |

ใช้ pretrained `yolov8n.pt` ใหม่ทุกการทดลอง และฝึกบน CPU เป็นค่าเริ่มต้น หากใช้ GPU ที่รองรับสามารถเปลี่ยน `device` ใน `detection/train.py` ได้

## ผลการทดลอง

เมื่อฝึกเสร็จ Ultralytics จะบันทึกกราฟ `results.png`, `results.csv`, กราฟ PR และ Confusion Matrix ลงใน `outputs/model_*/` ส่วนผลประเมิน Test และการเปรียบเทียบจะอยู่ใน `outputs/eval_*` และ `outputs/comparison.csv`, `comparison.png`

ค่า Precision, Recall และ mAP ต้องใช้ผลฝึกจริงก่อนจึงจะนำไปสรุปได้ เนื่องจาก Dataset มีเพียง 149 ภาพ ผลทดสอบอาจผันผวน

## แหล่งอ้างอิง

- Dataset: https://www.kaggle.com/datasets/tannergi/microcontroller-detection
- Ultralytics: https://docs.ultralytics.com/
- เอกสารประกอบบทเรียน Chapter 10: Object Detection II
