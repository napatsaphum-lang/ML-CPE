# NASA JPL GRACE Dataset - Data Preprocessing

### Overview

This project demonstrates Data Preprocessing techniques using the NASA JPL GRACE Monthly Land Water Storage Anomalies Dataset.

The workflow includes dataset exploration, data visualization, data cleaning, and feature engineering to prepare the dataset for Machine Learning applications.

### Dataset

The dataset contains monthly land water storage anomaly data from NASA JPL GRACE and GRACE-FO satellite missions.

Dataset: `GRACE_GRACE-FO_Months_RL06.csv`

Source: Kaggle - NASA JPL GRACE Monthly Land Water Storage Anomalies

### Tasks

- Dataset Exploration
  - Shape
  - Data Types
  - Summary Statistics
  - Missing Values
  - Duplicate Records
  - Class Distribution

- Data Visualization
  - Histogram
  - Correlation Heatmap

- Data Cleaning
  - Missing Value Handling
  - Duplicate Removal
  - Incorrect Data Correction
  - Data Type Conversion
  - Mean and Median Comparison

- Feature Engineering
  - Label Encoding
  - One-Hot Encoding

### Output Files

- `GRACE_Cleaned_Data.csv`
- `GRACE_Label_Encoded.csv`
- `GRACE_OneHot_Encoded.csv`

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

### ภาพรวม

งานนี้เป็นการศึกษาและประยุกต์ใช้กระบวนการ Data Preprocessing กับชุดข้อมูล NASA JPL GRACE Monthly Land Water Storage Anomalies เพื่อเตรียมข้อมูลให้อยู่ในรูปแบบที่เหมาะสมก่อนนำไปใช้กับ Machine Learning

### สิ่งที่ดำเนินการ

- สำรวจข้อมูลเบื้องต้น
  - ตรวจสอบ Shape
  - ตรวจสอบ Data Types
  - Summary Statistics
  - Missing Values
  - Duplicate Records
  - Class Distribution

- การแสดงผลข้อมูล
  - Histogram
  - Correlation Heatmap

- การทำความสะอาดข้อมูล
  - จัดการ Missing Values
  - ลบข้อมูลซ้ำ
  - แก้ไขข้อมูลที่ไม่ถูกต้อง
  - แปลงชนิดข้อมูล
  - เปรียบเทียบ Mean และ Median

- Feature Engineering
  - Label Encoding
  - One-Hot Encoding

### เครื่องมือที่ใช้

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

### Run

Install required libraries:

    pip install pandas numpy matplotlib scikit-learn

Run the program:

    python Data_preprocessing.py
