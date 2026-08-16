# LAB05 - Support Vector Machine (SVM)

## Overview

This laboratory applies **Support Vector Machine (SVM)** to classify the Iris dataset and compare the performance of three different kernel functions:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel

ใบงานนี้เป็นการทดลองใช้ SVM สำหรับการจำแนกข้อมูล และเปรียบเทียบประสิทธิภาพของ Kernel แต่ละประเภท

---

## Objectives

- Apply SVM for classification.
- Explore and preprocess the dataset.
- Standardize the input features before training.
- Train SVM models using Linear, Polynomial, and RBF kernels.
- Evaluate each model using Accuracy.
- Compare prediction results.

---

## Dataset

This laboratory uses the **Iris Dataset**, which contains 150 samples and four input features.

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length |
| `SepalWidthCm` | Sepal width |
| `PetalLengthCm` | Petal length |
| `PetalWidthCm` | Petal width |

The target variable is `Species`, which contains three classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

### Dataset Source

**Iris Species - Kaggle**  
https://www.kaggle.com/datasets/uciml/iris

---

## Project Structure

```text
LAB05/
├── images/
│   ├── svm_accuracy.png
│   ├── species_count.png
│   └── iris_scatter.png
├── .gitignore
├── Iris.csv
├── main.py
├── README.md
└── requirements.txt
```

---

## Workflow

```text
Iris Dataset
     ↓
Load & Explore Data
     ↓
Data Preprocessing
     ↓
Train / Test Split
     ↓
Standardization
     ↓
Linear | Polynomial | RBF
     ↓
Model Training
     ↓
Prediction
     ↓
Accuracy Comparison
```

ขั้นตอนการทำงานเริ่มจากการอ่านและตรวจสอบข้อมูล แบ่งข้อมูลเป็น Training และ Testing จากนั้นทำ Standardization ก่อนนำไป Train ด้วย SVM ทั้ง 3 Kernel

---

## Data Exploration

The dataset is checked for missing values and the number of samples in each species.

### Iris Species Distribution

![Iris Species Distribution](images/species_count.png)

Each Iris species contains 50 samples.

---

## Data Visualization

The scatter plot shows the relationship between **Petal Length** and **Petal Width** for each Iris species.

![Iris Dataset Scatter Plot](images/iris_scatter.png)

กราฟช่วยแสดงการกระจายตัวและความแตกต่างของข้อมูลแต่ละ Species ก่อนนำไปสร้าง Model

---

## Data Preprocessing

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The input features are standardized using `StandardScaler` before training.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## SVM Models

Three SVM kernels are used in this experiment:

```python
SVC(kernel="linear")
SVC(kernel="poly")
SVC(kernel="rbf")
```

Each model is trained and evaluated using the same training and testing data.

---

## Results

The performance of each model is evaluated using **Accuracy**.

| SVM Kernel | Accuracy |
|---|---:|
| Linear | 96.67% |
| Polynomial | 96.67% |
| RBF | 100.00% |

### Accuracy Comparison

![SVM Kernel Accuracy](images/svm_accuracy.png)

The **RBF Kernel achieved the highest accuracy at 100.00%** in this experiment.

---

## Prediction

The program compares the actual species with predictions from each SVM model.

```text
Actual              Linear              Polynomial          RBF
Iris-versicolor     Iris-versicolor     Iris-versicolor     Iris-versicolor
Iris-setosa         Iris-setosa         Iris-setosa         Iris-setosa
Iris-virginica      Iris-virginica      Iris-virginica      Iris-virginica
```

`Actual` represents the correct class, while `Linear`, `Polynomial`, and `RBF` represent the prediction from each model.

---

## Installation and Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main.py
```

Required libraries:

- pandas
- matplotlib
- scikit-learn

---

## Conclusion

Support Vector Machine was applied to classify the Iris dataset using **Linear, Polynomial, and RBF kernels**.

The results showed that the **RBF Kernel achieved the highest accuracy at 100.00%**, while Linear and Polynomial achieved 96.67%.

**สรุป:** จากการทดลองพบว่า SVM สามารถจำแนก Iris Dataset ได้อย่างมีประสิทธิภาพ และการเลือก Kernel ที่แตกต่างกันส่งผลต่อค่า Accuracy ของ Model

---

## Reference

**Iris Species - Kaggle**  
https://www.kaggle.com/datasets/uciml/iris

---

**LAB05 - Support Vector Machine (SVM)**
