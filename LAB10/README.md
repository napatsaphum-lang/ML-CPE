# LAB10 - Object Detection Using YOLOv8

## 1. บทนำ (Introduction)

การทดลอง LAB10 เป็นการนำ YOLOv8 มาใช้ตรวจจับและจำแนกอุปกรณ์ไมโครคอนโทรลเลอร์จากรูปภาพ โดยโมเดลสามารถแสดงตำแหน่งวัตถุ (Bounding Box) ชื่ออุปกรณ์ และค่าความมั่นใจ (Confidence Score)

ในการทดลองนี้จะเปรียบเทียบจำนวน Epochs, Image Size และ Batch Size ที่แตกต่างกัน เพื่อศึกษาผลต่อประสิทธิภาพการตรวจจับวัตถุ

## 2. Dataset ที่ใช้

**Dataset:** [Microcontroller Detection – Kaggle](https://www.kaggle.com/datasets/tannergi/microcontroller-detection)

Dataset ประกอบด้วยภาพอุปกรณ์อิเล็กทรอนิกส์ 4 ประเภท

| Class | อุปกรณ์ |
|---|---|
| 0 | ESP8266 |
| 1 | Heltec ESP32 LoRa |
| 2 | Arduino Nano |
| 3 | Raspberry Pi 3 |

ใช้ภาพทั้งหมด 149 ภาพ โดยข้อมูล Bounding Box เดิมอยู่ในรูปแบบ XML และนำมาแปลงเป็น YOLO TXT สำหรับการ Training

## 3. โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB10/
├── detection/
│   ├── main.py           # โปรแกรมหลัก
│   ├── data_loader.py    # เตรียม Dataset และแปลง XML
│   ├── train.py          # ฝึก YOLOv8
│   ├── evaluate.py       # ประเมิน Precision, Recall, mAP
│   ├── image_test.py     # ทดสอบรูปภาพ
│   ├── video_test.py     # ทดสอบวิดีโอ
│   └── realtime_test.py  # ทดสอบ Webcam
│
├── dataset/               # ข้อมูล Train / Val / Test
├── outputs/               # โมเดลและกราฟผลการทดลอง
├── requirements.txt
└── README.md
```

## 4. ขั้นตอนการทดลอง (Methodology)

1. โหลด Dataset จาก Kaggle และแปลง Annotation จาก XML เป็น YOLO
2. แบ่งข้อมูลเป็น Training, Validation และ Testing
3. โหลด Pretrained YOLOv8n Model
4. ฝึกโมเดลโดยปรับ Epochs, Image Size และ Batch Size
5. ประเมินผลด้วย Precision, Recall และ mAP
6. ทดสอบการตรวจจับผ่านรูปภาพ วิดีโอ และ Webcam

### รูปแบบการทดลอง

| Model | Epochs | Image Size | Batch |
|---|---:|---:|---:|
| A | 10 | 416 | 8 |
| B | 20 | 416 | 8 |
| C | 30 | 416 | 8 |
| D | 20 | 640 | 8 |
| E | 20 | 416 | 16 |

## 5. วิธีรันโปรแกรม (How to Run)

ติดตั้ง Libraries:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ทดสอบการทำงานเบื้องต้น:

```powershell
.\.venv\Scripts\python.exe -m detection.main --quick
```

ฝึกโมเดลทั้งหมด:

```powershell
.\.venv\Scripts\python.exe -m detection.main
```

### ทดสอบ Object Detection

**ตรวจจับจากรูปภาพ**

```powershell
.\.venv\Scripts\python.exe -m detection.image_test
```

**ตรวจจับจากวิดีโอ**

```powershell
.\.venv\Scripts\python.exe -m detection.video_test
```

**ตรวจจับผ่าน Webcam**

```powershell
.\.venv\Scripts\python.exe -m detection.realtime_test
```

## 6. ผลการทดลอง (Experimental Results)

### 6.1 Training และ Validation Results

กราฟแสดงการเปลี่ยนแปลงของ Loss, Precision, Recall และ mAP ระหว่างการฝึกโมเดล

**Model A – Training Results**

![Model A Training](outputs/model_A/results.png)

**Model B – Training Results**

![Model B Training](outputs/model_B/results.png)

**Model C – Training Results**

![Model C Training](outputs/model_C/results.png)

### 6.2 Model Comparison

กราฟเปรียบเทียบประสิทธิภาพการตรวจจับของแต่ละโมเดล

![Model Comparison](outputs/comparison.png)

### 6.3 Confusion Matrix

ใช้ตรวจสอบการจำแนกอุปกรณ์แต่ละประเภทว่าถูกต้องหรือผิดพลาดมากน้อยเพียงใด

![Confusion Matrix](outputs/model_B/confusion_matrix.png)

### 6.4 Object Detection Results

แสดงผลตรวจจับไมโครคอนโทรลเลอร์ด้วย Bounding Box พร้อม Class และ Confidence Score

*เพิ่มภาพผลการตรวจจับจริงจากโฟลเดอร์ outputs หลังการทดสอบ*

## 7. วิเคราะห์ผลการทดลอง (Discussion)

การทดลองนี้เปรียบเทียบปัจจัยหลัก 3 อย่าง ได้แก่

- **Epochs:** ศึกษาว่าการเพิ่มจำนวนรอบฝึกส่งผลต่อ Precision, Recall และ mAP อย่างไร
- **Image Size:** เปรียบเทียบความละเอียดของภาพว่ามีผลต่อการตรวจจับวัตถุหรือไม่
- **Batch Size:** ศึกษาผลของจำนวนภาพที่ใช้ในแต่ละรอบการปรับน้ำหนักโมเดล

เนื่องจาก Dataset มีจำนวนภาพไม่มาก ผลการตรวจจับอาจเปลี่ยนแปลงได้ตามการแบ่งข้อมูลและการฝึกโมเดล จึงต้องพิจารณาค่า Validation Loss และ mAP ร่วมกัน

## 8. สรุป (Conclusion)

การทดลอง LAB10 เป็นการศึกษาการตรวจจับวัตถุด้วย YOLOv8 โดยนำมาใช้กับอุปกรณ์ไมโครคอนโทรลเลอร์ 4 ประเภท

จากการทดลองทำให้เข้าใจขั้นตอนการเตรียม Dataset การฝึกโมเดล Object Detection และการประเมินประสิทธิภาพด้วย Precision, Recall และ mAP รวมถึงแนวทางนำโมเดลไปใช้ตรวจจับวัตถุผ่านรูปภาพ วิดีโอ และ Webcam

## 9. แหล่งอ้างอิง (References)

- [Microcontroller Detection Dataset – Kaggle](https://www.kaggle.com/datasets/tannergi/microcontroller-detection)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- เอกสารประกอบการเรียน Machine Learning Chapter 10: Object Detection II
