# LAB04 : K-Nearest Neighbors (KNN)

## Introduction

โปรเจกต์นี้เป็นส่วนหนึ่งของรายวิชา **Machine Learning** มีวัตถุประสงค์เพื่อศึกษาหลักการทำงานของอัลกอริทึม **K-Nearest Neighbors (KNN)** สำหรับการจำแนกข้อมูล (Classification)

ในการทดลองครั้งนี้เลือกใช้ **Iris Species Dataset** จาก Kaggle โดยนำข้อมูลมาวิเคราะห์ เตรียมข้อมูล ปรับข้อมูลให้อยู่ในมาตรฐานเดียวกัน (Standardization) และทดลองสร้างโมเดล KNN ด้วยค่า **k = 3, 5 และ 7** จากนั้นเปรียบเทียบผลลัพธ์เพื่อเลือกค่า k ที่เหมาะสมที่สุด

---

# Objectives

- ศึกษาหลักการทำงานของอัลกอริทึม KNN
- ทดลองสร้างโมเดลจำแนกข้อมูลด้วย KNN
- เปรียบเทียบผลลัพธ์ของค่า k ที่แตกต่างกัน
- ประเมินประสิทธิภาพของโมเดลด้วย Accuracy
- วิเคราะห์ผลการทดลองและเลือกค่า k ที่เหมาะสม

---

# Dataset

ในการทดลองใช้ชุดข้อมูล **Iris Species Dataset**

Dataset นี้เป็นข้อมูลของดอก Iris จำนวน 150 ตัวอย่าง ประกอบด้วยดอกไม้ 3 ชนิด ได้แก่

- Setosa
- Versicolor
- Virginica

โดยแต่ละตัวอย่างจะมีข้อมูลทั้งหมด 4 คุณลักษณะ ได้แก่

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

ใช้คอลัมน์ **Species** เป็นคำตอบ (Target)

**Dataset Source**

Kaggle. *Iris Species Dataset*

https://www.kaggle.com/datasets/uciml/iris

---

# Development Tools

โปรแกรมที่ใช้ในการพัฒนา

- Python
- Visual Studio Code

Library ที่ใช้

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Data Preprocessing

ก่อนสร้างโมเดลได้มีการเตรียมข้อมูล ดังนี้

1. โหลดข้อมูลจากไฟล์ iris.csv
2. ตรวจสอบข้อมูลเบื้องต้น
3. ตรวจสอบ Missing Value
4. ลบข้อมูลซ้ำ
5. แบ่งข้อมูลเป็น Training และ Testing
6. ปรับข้อมูลด้วย StandardScaler

---

# Machine Learning

## Clustering

ทดลองจัดกลุ่มข้อมูลด้วย

- K-Means Clustering

เพื่อดูการกระจายตัวของข้อมูลก่อนนำไปสร้างโมเดล

---

## Classification

ใช้

**K-Nearest Neighbors (KNN)**

ทดลองทั้งหมด 3 ค่า คือ

- k = 3
- k = 5
- k = 7

---

# Evaluation

การประเมินผลใช้

- Accuracy
- Classification Report
- Confusion Matrix

จากนั้นเลือกค่า k ที่ให้ Accuracy สูงที่สุด

---

# Project Structure

```text
LAB04
│
├── LAB4.py
├── iris.csv
├── README.md
├── requirements.txt
├── .gitignore
└── images
    ├── iris_dataset.png
    ├── kmeans_clustering.png
    ├── knn_accuracy.png
    └── confusion_matrix.png
```

---

# Installation

ติดตั้ง Library

```bash
pip install -r requirements.txt
```

---

# Run

รันโปรแกรม

```bash
python LAB4.py
```

---

# Results

## Iris Dataset

![Iris Dataset](images/iris_dataset.png)

---

## K-Means Clustering

![K-Means Clustering](images/kmeans_clustering.png)

---

## Accuracy Comparison

![Accuracy Comparison](images/knn_accuracy.png)

---

## Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

# Discussion

จากการทดลองพบว่า ค่า **k** ที่แตกต่างกันส่งผลต่อความแม่นยำของโมเดล โดยค่า **Best k** ให้ค่า Accuracy สูงที่สุดในการทดลอง

การปรับข้อมูลด้วย StandardScaler ช่วยให้ทุก Feature อยู่ในช่วงค่าที่ใกล้เคียงกัน ทำให้การคำนวณระยะทางของ KNN มีประสิทธิภาพมากขึ้น และช่วยให้ผลการจำแนกข้อมูลมีความแม่นยำสูงขึ้น

---

# Conclusion

จากการทดลองสามารถนำอัลกอริทึม K-Nearest Neighbors (KNN) มาใช้ในการจำแนกชนิดของดอก Iris ได้อย่างมีประสิทธิภาพ โดยการเลือกค่า **k** ที่เหมาะสมช่วยให้โมเดลมี Accuracy ที่ดี นอกจากนี้การเตรียมข้อมูลก่อนการสร้างโมเดล เช่น การตรวจสอบข้อมูลและการทำ Standardization ยังเป็นขั้นตอนสำคัญที่ช่วยเพิ่มประสิทธิภาพของการจำแนกข้อมูล
