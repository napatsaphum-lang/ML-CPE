# LAB08 - Deep Convolutional Neural Network (DCNN)

**รายวิชา:** Machine Learning (04-624-201)  
**หัวข้อ:** Deep Convolutional Neural Network และการประยุกต์ใช้งาน  
**Dataset:** Intel Image Classification  
**เครื่องมือ:** Python, TensorFlow, Keras, VS Code

---

## 1. บทนำ (Introduction)

การทดลอง LAB08 เป็นการศึกษาการทำงานของ Deep Convolutional Neural Network (DCNN) ซึ่งเป็นโครงข่ายประสาทเทียมที่นำมาใช้ในการจำแนกรูปภาพ โดยพัฒนาต่อยอดจาก Convolutional Neural Network (CNN) ด้วยการเพิ่มจำนวน Convolutional Layers เพื่อให้โมเดลสามารถเรียนรู้ลักษณะของภาพได้ละเอียดและซับซ้อนมากขึ้น

ในการทดลองนี้ได้เลือกใช้ Intel Image Classification Dataset จาก Kaggle เนื่องจากเป็นชุดข้อมูลรูปภาพสถานที่และธรรมชาติที่มีลักษณะแตกต่างกัน เหมาะสำหรับนำมาทดลองจำแนกภาพด้วย DCNN

การทดลองจะเปรียบเทียบผลการทำงานของโมเดลที่มีจำนวน Convolutional Layers, Dense Neurons และ Epochs แตกต่างกัน เพื่อศึกษาว่าการปรับโครงสร้างและจำนวนรอบในการฝึกส่งผลต่อความแม่นยำของโมเดลอย่างไร

## 2. วัตถุประสงค์ (Objectives)

1. ศึกษาหลักการทำงานของ Deep Convolutional Neural Network (DCNN)
2. ฝึกการเตรียมข้อมูลรูปภาพสำหรับ Machine Learning
3. สร้างโมเดล DCNN เพื่อจำแนกประเภทรูปภาพ
4. เปรียบเทียบผลของจำนวน Convolutional Layers และ Dense Neurons
5. ทดลองจำนวน Epochs ที่แตกต่างกัน
6. วิเคราะห์ผล Training Accuracy, Validation Accuracy และ Loss
7. ทดสอบความสามารถของโมเดลในการทำนายรูปภาพที่ไม่เคยใช้ฝึก

## 3. Dataset ที่ใช้

**Dataset:** Intel Image Classification  
**แหล่งข้อมูล:** [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

Dataset นี้ประกอบด้วยภาพสถานที่และธรรมชาติทั้งหมด 6 ประเภท ได้แก่

| Class | ประเภทภาพ |
|---|---|
| Buildings | อาคารและสิ่งก่อสร้าง |
| Forest | ป่าไม้ |
| Glacier | ธารน้ำแข็ง |
| Mountain | ภูเขา |
| Sea | ทะเล |
| Street | ถนน |

ข้อมูลแบ่งเป็นชุด Training จำนวน 14,034 ภาพ และ Testing จำนวน 3,000 ภาพ นอกจากนี้ยังมีรูปภาพสำหรับทดลอง Prediction อีก 7,301 ภาพ

เลือกใช้ Dataset นี้เพราะภาพมีความหลากหลาย ทั้งสี รูปร่าง และองค์ประกอบของสถานที่ ทำให้สามารถศึกษาประสิทธิภาพการจำแนกภาพของ DCNN ได้

## 4. โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB08/
│
├── classification/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── dcnn_model.py
│   ├── evaluate.py
│   ├── test_model.py
│   │
│   └── outputs/
│       ├── classes.json
│       ├── best_model.keras
│       ├── comparison.csv
│       ├── comparison.png
│       ├── history_A.png
│       ├── history_B.png
│       ├── confusion_matrix.png
│       ├── classification_report.txt
│       └── predictions.png
│
├── requirements.txt
└── README.md
```

**หน้าที่ของไฟล์แต่ละส่วน**

- `main.py` เป็นไฟล์หลักสำหรับควบคุมการทำงานของโปรแกรม
- `data_loader.py` ใช้โหลด Dataset และจัดเตรียมโฟลเดอร์รูปภาพ
- `preprocessing.py` กำหนดค่าการเตรียมรูปภาพก่อนนำเข้าโมเดล
- `split_data.py` แบ่งข้อมูลสำหรับ Training, Validation และ Testing
- `dcnn_model.py` ใช้สร้างโครงสร้างและกำหนดการฝึกโมเดล DCNN
- `evaluate.py` ใช้ประเมินความแม่นยำและบันทึกกราฟผลการทดลอง
- `test_model.py` ใช้ทดสอบโมเดลด้วยรูปภาพ
- `outputs/` ใช้จัดเก็บโมเดลและผลลัพธ์จากการทดลอง

การแยกไฟล์ตามหน้าที่ช่วยให้สามารถตรวจสอบและแก้ไขโค้ดแต่ละส่วนได้ง่ายขึ้น โดยไม่ต้องรวมคำสั่งทั้งหมดไว้ในไฟล์เดียว

## 5. ขั้นตอนการทำงาน (Methodology)

### 5.1 การเตรียมข้อมูล (Data Preprocessing)

เริ่มจากนำ Dataset มาโหลดเข้าสู่โปรแกรม จากนั้นเตรียมข้อมูลรูปภาพให้อยู่ในรูปแบบที่โมเดลสามารถประมวลผลได้

ขั้นตอนหลักประกอบด้วย

1. โหลดรูปภาพจาก Dataset
2. ปรับขนาดรูปภาพให้เท่ากัน
3. ปรับค่าพิกเซลให้อยู่ในช่วง 0–1
4. แยกข้อมูล Training, Validation และ Testing
5. จัดข้อมูลเป็น Batch สำหรับใช้ในการ Training

การปรับค่าพิกเซลช่วยให้ข้อมูลมีสเกลใกล้เคียงกันและเหมาะสมกับการฝึกโครงข่ายประสาทเทียม

### 5.2 การสร้างโมเดล DCNN

โมเดล DCNN ที่ใช้ในการทดลองประกอบด้วย Layer หลักดังนี้

| Layer | หน้าที่ |
|---|---|
| Conv2D | สกัดลักษณะสำคัญจากรูปภาพ |
| ReLU | ช่วยให้โมเดลเรียนรู้ข้อมูลที่มีความซับซ้อน |
| MaxPooling2D | ลดขนาดข้อมูลและ Feature Maps |
| Flatten | แปลงข้อมูลหลายมิติให้เป็นเวกเตอร์ |
| Dense | นำ Feature มาประมวลผลเพื่อจำแนกภาพ |
| Dropout | ช่วยลดโอกาสเกิด Overfitting |
| Softmax | คำนวณความน่าจะเป็นของภาพในแต่ละ Class |

เนื่องจากมีข้อมูลทั้งหมด 6 Class จึงกำหนด Output Layer ให้สามารถจำแนกผลลัพธ์ได้ 6 ประเภท

### 5.3 การออกแบบการทดลอง (Experiments)

ในการทดลองได้กำหนดโมเดลทั้งหมด 6 รูปแบบ โดยเปลี่ยนจำนวน Convolutional Layers, Dense Neurons และ Epochs

| Model | Conv Layers | Dense Neurons | Epochs |
|---|---:|---:|---:|
| Model A | 2 | 128 | 10 |
| Model B | 3 | 128 | 10 |
| Model C | 4 | 128 | 10 |
| Model D | 3 | 128 | 20 |
| Model E | 3 | 128 | 30 |
| Model F | 3 | 256 | 10 |

การทดลองแบ่งเป็น 3 ส่วน ได้แก่

**การเปรียบเทียบจำนวน Convolutional Layers**

เปรียบเทียบ Model A, B และ C โดยใช้จำนวน Epochs เท่ากัน เพื่อศึกษาว่าการเพิ่มจำนวน Convolutional Layers ส่งผลต่อ Accuracy อย่างไร

**การเปรียบเทียบจำนวน Epochs**

เปรียบเทียบ Model B, D และ E ซึ่งใช้โครงสร้างเดียวกัน แต่ฝึกด้วยจำนวน Epochs ที่แตกต่างกัน เพื่อสังเกตการเปลี่ยนแปลงของ Accuracy และ Loss

**การเปรียบเทียบจำนวน Dense Neurons**

เปรียบเทียบ Model B และ F ซึ่งมีจำนวน Convolutional Layers และ Epochs เท่ากัน แต่ใช้จำนวน Dense Neurons ต่างกัน

## 6. การติดตั้งและรันโปรแกรม (How to Run)

โปรเจกต์นี้พัฒนาโดยใช้ Python 3.11 และทำงานผ่าน VS Code

**1. สร้าง Virtual Environment**

```powershell
py -3.11 -m venv .venv
```

**2. ติดตั้ง Libraries**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. เตรียม Dataset**

ดาวน์โหลดไฟล์จาก Kaggle แล้วนำไฟล์ `archive (7).zip` มาวางไว้ในโฟลเดอร์หลักของโปรเจกต์

**4. ทดสอบโปรแกรมแบบรวดเร็ว**

```powershell
python -m classification.main --quick
```

**5. เริ่มฝึกโมเดลทั้งหมด**

```powershell
python -m classification.main
```

**6. ทดสอบการทำนาย**

```powershell
python -m classification.test_model
```

เมื่อโปรแกรมทำงานเสร็จ ผลการทดลองจะถูกบันทึกไว้ในโฟลเดอร์ `classification/outputs/`

## 7. ผลการทดลอง (Experimental Results)

ผลการทดลองจะประเมินจาก Accuracy และ Loss ของแต่ละโมเดล โดยนำผลลัพธ์มาเปรียบเทียบเพื่อหาว่าโครงสร้างใดมีประสิทธิภาพเหมาะสมกับ Dataset ที่ใช้

### 7.1 การเปรียบเทียบ Accuracy

ผลลัพธ์จะถูกบันทึกไว้ในไฟล์ `comparison.csv` และแสดงกราฟผ่าน `comparison.png`

| Model | Epochs | Test Accuracy |
|---|---:|---|
| A | 10 | รอผลการทดลอง |
| B | 10 | รอผลการทดลอง |
| C | 10 | รอผลการทดลอง |
| D | 20 | รอผลการทดลอง |
| E | 30 | รอผลการทดลอง |
| F | 10 | รอผลการทดลอง |

### 7.2 Training และ Validation Accuracy/Loss

กราฟ Training History ใช้ตรวจสอบการเปลี่ยนแปลงของ Accuracy และ Loss ในแต่ละ Epoch

สิ่งที่ต้องสังเกตคือ หาก Training Accuracy เพิ่มขึ้น แต่ Validation Accuracy ไม่เพิ่มขึ้นตาม หรือ Validation Loss กลับสูงขึ้น อาจแสดงว่าโมเดลเกิด Overfitting

### 7.3 Confusion Matrix

ใช้ตรวจสอบการทำนายของโมเดลว่าแต่ละ Class ถูกจำแนกได้ถูกต้องมากน้อยเพียงใด รวมถึงดูว่าโมเดลมีการทำนายสับสนระหว่างภาพประเภทใดบ้าง

### 7.4 Prediction Results

นำภาพจากชุด Testing มาทำนาย แล้วเปรียบเทียบผลที่โมเดลทำนายกับ Class จริง เพื่อทดสอบว่าโมเดลสามารถจำแนกภาพที่ไม่เคยใช้ฝึกได้หรือไม่

## 8. การวิเคราะห์ผลการทดลอง (Discussion)

การวิเคราะห์ผลจะพิจารณา 3 ประเด็นหลัก

**1. จำนวน Convolutional Layers**

การเพิ่ม Convolutional Layers ช่วยเพิ่มความสามารถในการเรียนรู้ลักษณะภาพที่ซับซ้อน แต่ไม่ได้หมายความว่าจะทำให้ Accuracy สูงขึ้นเสมอไป เพราะโมเดลที่ลึกขึ้นอาจต้องใช้ข้อมูลและเวลาฝึกเพิ่มขึ้น

**2. จำนวน Epochs**

การเพิ่ม Epochs ทำให้โมเดลมีโอกาสเรียนรู้ข้อมูลมากขึ้น แต่หากฝึกมากเกินไปอาจเกิด Overfitting จึงต้องพิจารณา Validation Accuracy และ Validation Loss ร่วมด้วย

**3. จำนวน Dense Neurons**

การเพิ่ม Dense Neurons ทำให้โมเดลมีพารามิเตอร์ในการเรียนรู้มากขึ้น แต่ก็อาจเพิ่มความซับซ้อนและระยะเวลา Training โดยไม่จำเป็นต้องทำให้ผลการจำแนกดีขึ้นเสมอไป

หลังจากได้ผลจริง จะนำค่าของแต่ละโมเดลมาเปรียบเทียบเพื่อพิจารณาว่า Configuration ใดเหมาะกับ Dataset มากที่สุด

## 9. สรุป (Conclusion)

การทดลอง LAB08 เป็นการนำ Deep Convolutional Neural Network มาประยุกต์ใช้กับการจำแนกภาพสถานที่และธรรมชาติจาก Intel Image Classification Dataset โดยสร้างโมเดลที่มีโครงสร้างและจำนวน Epochs แตกต่างกัน

สิ่งสำคัญของการทดลองนี้คือการศึกษาความสัมพันธ์ระหว่างความซับซ้อนของโมเดล จำนวนรอบในการฝึก และประสิทธิภาพการทำนาย ไม่ใช่การพิจารณา Accuracy เพียงอย่างเดียว

ผลการทดลองจริงจะถูกนำมาใช้สรุปว่าโมเดลใดเหมาะสมที่สุด รวมถึงวิเคราะห์ข้อจำกัดและแนวทางในการปรับปรุงโมเดลต่อไป

---

## 10. References (แหล่งอ้างอิง)

**Dataset ที่ใช้ในการทดลอง**

- **ชื่อ Dataset:** Intel Image Classification
- **ผู้เผยแพร่:** puneet6060
- **แหล่งข้อมูล:** Kaggle
- **ประเภทข้อมูล:** รูปภาพสำหรับการจำแนกสถานที่และธรรมชาติ 6 ประเภท
- **URL:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification

**เอกสารและเครื่องมือที่ใช้**

1. Puneet6060. *Intel Image Classification*. Kaggle. https://www.kaggle.com/datasets/puneet6060/intel-image-classification
2. TensorFlow. *TensorFlow Documentation*. https://www.tensorflow.org/
3. Keras. *Keras Documentation*. https://keras.io/
4. Anuruk Prommakhot, Ph.D. *Machine Learning – Chapter 8: Deep Convolutional Neural Network (DCNN)*. เอกสารประกอบการเรียนรายวิชา 04-624-201 มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี
