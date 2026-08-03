LAB 3: Regression & Classification

โปรเจกต์นี้ใช้ข้อมูล age_gender.csv เพื่อทดลองสร้างโมเดล Machine Learning สำหรับ

Simple Linear Regression

Multiple Linear Regression

Age Prediction

Principal Component Analysis (PCA)

Logistic Regression

Gender Classification

การประเมิน Accuracy, Precision, Recall, F1-score, ROC Curve และ AUC

โครงสร้างไฟล์

ML-LAB3/
├── LAB3.py
├── age_gender.csv
├── requirements.txt
├── README.md
└── .gitignore

ไฟล์ age_gender.csv ไม่ได้รวมอยู่ในโปรเจกต์ ZIP ให้แตกไฟล์ Dataset แล้วนำ CSV มาไว้ในโฟลเดอร์เดียวกับ LAB3.py

วิธีเปิดใน VS Code

แตกไฟล์ ML-LAB3.zip

แตกไฟล์ Dataset archive (2).zip

คัดลอก age_gender.csv มาไว้ในโฟลเดอร์ ML-LAB3

เปิดโฟลเดอร์ ML-LAB3 ด้วย VS Code

เปิด Terminal ใน VS Code

สร้าง Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS หรือ Linux:

python3 -m venv venv
source venv/bin/activate

ติดตั้ง Library

pip install -r requirements.txt

รันโปรแกรม

python LAB3.py

ผลลัพธ์

โปรแกรมจะแสดงค่าประเมินผลใน Terminal และสร้างกราฟดังนี้

simple_linear_regression.png

multiple_linear_regression.png

roc_curve.png

หมายเหตุ

โปรแกรมกำหนด SAMPLE_SIZE = 6000 เพื่อให้สามารถรันบนคอมพิวเตอร์ทั่วไปได้เร็วขึ้น หากเครื่องมี RAM เพียงพอ สามารถเพิ่มจำนวนได้ภายในไฟล์ LAB3.py
