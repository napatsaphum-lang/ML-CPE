# LAB05 - Support Vector Machine

ใบงานที่ 5 การประยุกต์ใช้งาน Support Vector Machine (SVM)

## Dataset

ใช้ Iris Dataset สำหรับทดลองการจำแนกประเภทดอก Iris

Features ที่ใช้มี 4 ตัว

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target ที่ใช้คือ Species

Species แบ่งออกเป็น 3 ประเภท

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## การทำงานของโปรแกรม

1. อ่านข้อมูลจาก Iris.csv
2. ตรวจสอบข้อมูลเบื้องต้น
3. เลือก Features และ Target
4. แบ่ง Training และ Testing Data
5. ปรับข้อมูลด้วย StandardScaler
6. สร้าง SVM แบบ Linear Kernel
7. สร้าง SVM แบบ Polynomial Kernel
8. สร้าง SVM แบบ RBF Kernel
9. ทำนายข้อมูล
10. เปรียบเทียบค่า Accuracy

## Library

- pandas
- matplotlib
- scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Output

โปรแกรมจะแสดง

- Accuracy ของ Linear Kernel
- Accuracy ของ Polynomial Kernel
- Accuracy ของ RBF Kernel
- Kernel ที่มี Accuracy สูงที่สุด
- Prediction Results
- กราฟเปรียบเทียบ Accuracy