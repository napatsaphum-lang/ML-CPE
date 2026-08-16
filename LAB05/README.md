# LAB05 - Support Vector Machine (SVM)

## Support Vector Machine and SVM Applications

This laboratory applies the **Support Vector Machine (SVM)** algorithm to classify the Iris dataset and compare the performance of different SVM kernel functions.

ใบงานนี้เป็นการทดลองใช้งาน Support Vector Machine (SVM) สำหรับการจำแนกข้อมูล โดยใช้ Iris Dataset เป็นชุดข้อมูลในการทดลอง

---

## Objectives

The objectives of this laboratory are:

- Apply Support Vector Machine (SVM) for classification.
- Select and load a dataset.
- Explore and preprocess the dataset.
- Standardize the input features before training.
- Train SVM models using three different kernel functions.
- Compare the performance of Linear, Polynomial, and RBF kernels.
- Evaluate each model using Accuracy.
- Display prediction results from the testing dataset.

**วัตถุประสงค์:** เพื่อศึกษาและทดลองการทำงานของ SVM สำหรับการจำแนกข้อมูล และเปรียบเทียบประสิทธิภาพของ Kernel แต่ละประเภท

---

## Dataset

This laboratory uses the **Iris Dataset** for classification.

The dataset contains **150 samples** of Iris flowers and four input features:

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |

The target variable is `Species`.

The dataset contains three Iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

ข้อมูล Iris Dataset ถูกนำมาใช้สำหรับทดลองการจำแนกสายพันธุ์ดอก Iris โดยพิจารณาจากลักษณะของ Sepal และ Petal

---

## Dataset Source

The dataset used in this laboratory was obtained from Kaggle.

**Dataset:** Iris Species  
**Source:** Kaggle - UCI Machine Learning

[Iris Species - Kaggle](https://www.kaggle.com/datasets/uciml/iris)

The dataset is used for educational purposes to study and experiment with classification using Support Vector Machine.

---

## Project Structure

```text
LAB05/
├── images/
│   ├── svm_accuracy.png
│   ├── species_count.png
│   └── iris_scatter.png
│
├── .gitignore
├── Iris.csv
├── main.py
├── README.md
└── requirements.txt
```

---

## Workflow

The program follows the workflow below:

```text
Iris Dataset
      |
      v
Load Dataset
      |
      v
Explore Dataset
      |
      v
Check Missing Values
      |
      v
Select Features and Target
      |
      v
Train / Test Split
      |
      v
Standardization
      |
      v
+------------+------------+------------+
|   Linear   | Polynomial |    RBF     |
+------------+------------+------------+
      |            |            |
      v            v            v
    Train        Train        Train
      |            |            |
      v            v            v
   Predict      Predict      Predict
      |            |            |
      v            v            v
  Accuracy     Accuracy     Accuracy
       \           |           /
        \          |          /
         +---------+---------+
                   |
                   v
             Compare Results
```

ขั้นตอนการทำงานเริ่มจากการอ่านและตรวจสอบข้อมูล จากนั้นแบ่งข้อมูลสำหรับ Training และ Testing ก่อนปรับมาตรฐานข้อมูล และนำไป Train ด้วย SVM ทั้ง 3 Kernel

---

## Data Exploration

Before training the SVM models, the dataset is explored by checking:

- First five rows of the dataset
- Number of rows and columns
- Missing values
- Number of samples in each species

This step is used to understand the dataset before preprocessing and model training.

### Iris Species Distribution

The following graph shows the number of samples for each Iris species.

![Number of Iris Species](images/species_count.png)

กราฟแสดงจำนวนข้อมูลของ Iris แต่ละ Species ที่ใช้ในการทดลอง

---

## Data Visualization

Petal Length and Petal Width are used to visualize the distribution of the three Iris species.

![Iris Dataset Scatter Plot](images/iris_scatter.png)

The scatter plot shows the relationship between **Petal Length** and **Petal Width**.

กราฟใช้สำหรับดูการกระจายตัวของข้อมูลแต่ละ Species และช่วยให้เห็นความแตกต่างของข้อมูลเบื้องต้นก่อนนำไปสร้าง Model

---

## Data Preprocessing

The input features used in this experiment are:

```text
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

The target variable is:

```text
Species
```

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The data is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

```text
Training Data : 120 samples
Testing Data  : 30 samples
```

---

## Standardization

Before training the SVM models, the input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization scales the input features before they are used for model training.

การทำ Standardization เป็นขั้นตอนการปรับข้อมูล Feature ให้อยู่ในมาตรฐานเดียวกันก่อนนำข้อมูลไป Train Model

---

## SVM Models

Three SVM kernel functions are used in this experiment.

### 1. Linear Kernel

```python
linear_model = SVC(kernel="linear")
```

### 2. Polynomial Kernel

```python
poly_model = SVC(kernel="poly")
```

### 3. RBF Kernel

```python
rbf_model = SVC(kernel="rbf")
```

Each model is trained and tested using the same training and testing datasets so that the results can be compared.

---

## Model Training

The models are trained using the training dataset.

```python
linear_model.fit(X_train, y_train)
poly_model.fit(X_train, y_train)
rbf_model.fit(X_train, y_train)
```

After training, each model predicts the species from the testing dataset.

```python
linear_pred = linear_model.predict(X_test)
poly_pred = poly_model.predict(X_test)
rbf_pred = rbf_model.predict(X_test)
```

---

## Model Evaluation

The performance of each SVM model is evaluated using **Accuracy**.

Accuracy is calculated by comparing the actual species with the predicted species.

```python
linear_acc = accuracy_score(y_test, linear_pred)
poly_acc = accuracy_score(y_test, poly_pred)
rbf_acc = accuracy_score(y_test, rbf_pred)
```

---

## Accuracy Results

The experimental results are:

| SVM Kernel | Accuracy |
|---|---:|
| Linear | 96.67% |
| Polynomial | 96.67% |
| RBF | 100.00% |

### SVM Kernel Accuracy Comparison

![SVM Kernel Accuracy](images/svm_accuracy.png)

The experimental results show that the **RBF Kernel achieved the highest accuracy at 100.00%**.

จากผลการทดลองพบว่า RBF Kernel ให้ค่า Accuracy สูงที่สุดสำหรับชุดข้อมูลและการแบ่ง Training/Testing ที่ใช้ในการทดลองครั้งนี้

---

## Prediction Results

The program predicts the Iris species from the testing dataset using all three SVM models.

The prediction results contain:

| Column | Description |
|---|---|
| `Actual` | Actual Iris species |
| `Linear` | Prediction from Linear Kernel |
| `Polynomial` | Prediction from Polynomial Kernel |
| `RBF` | Prediction from RBF Kernel |

Example output:

```text
===== Prediction Results =====

Actual              Linear              Polynomial          RBF
Iris-versicolor     Iris-versicolor     Iris-versicolor     Iris-versicolor
Iris-setosa         Iris-setosa         Iris-setosa         Iris-setosa
Iris-virginica      Iris-virginica      Iris-virginica      Iris-virginica
```

The `Actual` column represents the actual class from the dataset, while the other columns represent predictions from each SVM model.

ผล Prediction สามารถใช้เปรียบเทียบค่าจริงกับค่าที่ Model แต่ละ Kernel ทำนายได้

---

## Result Summary

The accuracy results from the three SVM kernels are:

```text
Linear Kernel     : 96.67%
Polynomial Kernel : 96.67%
RBF Kernel        : 100.00%
```

The best performing model is:

```text
Best Kernel : RBF
Accuracy    : 100.00%
```

---

## Generated Graphs

The program generates and saves three graphs in the `images` directory.

### 1. Iris Species Distribution

![Number of Iris Species](images/species_count.png)

This graph shows the number of samples for each Iris species.

### 2. Iris Dataset Scatter Plot

![Iris Dataset Scatter Plot](images/iris_scatter.png)

This graph shows the distribution of Iris species based on Petal Length and Petal Width.

### 3. SVM Kernel Accuracy

![SVM Kernel Accuracy](images/svm_accuracy.png)

This graph compares the accuracy of Linear, Polynomial, and RBF kernels.

---

## Libraries

The following Python libraries are used in this laboratory:

- pandas
- matplotlib
- scikit-learn

The required libraries are listed in:

```text
requirements.txt
```

---

## Installation

Open the terminal in the `LAB05` directory.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Program

Run the program using:

```bash
python main.py
```

The program will display:

- Dataset information
- Missing values
- Number of each Iris species
- Training and testing data size
- Accuracy scores for each SVM kernel
- Best performing kernel
- Prediction results
- Result graphs

The generated graphs will be automatically saved in the `images` directory.

---

## Output Files

After running the program, the following graph files are generated:

```text
images/
├── svm_accuracy.png
├── species_count.png
└── iris_scatter.png
```

---

## Conclusion

In this laboratory, Support Vector Machine was applied to classify the Iris dataset using three different kernel functions: **Linear, Polynomial, and RBF**.

The dataset was explored and preprocessed before training. The input features were standardized using `StandardScaler`, and the dataset was divided into 80% training data and 20% testing data.

The experimental results were:

- **Linear Kernel: 96.67%**
- **Polynomial Kernel: 96.67%**
- **RBF Kernel: 100.00%**

The **RBF Kernel achieved the highest accuracy at 100.00%** for the dataset and train/test split used in this experiment.

**สรุปผลการทดลอง:** จากการทดลอง SVM ทั้ง 3 Kernel พบว่า RBF Kernel ให้ค่า Accuracy สูงที่สุดที่ 100.00% ส่วน Linear และ Polynomial ให้ค่า Accuracy เท่ากันที่ 96.67% แสดงให้เห็นว่าการเลือก Kernel ที่เหมาะสมสามารถส่งผลต่อประสิทธิภาพในการจำแนกข้อมูล

---

## References

### Dataset

**Iris Species - UCI Machine Learning**

Kaggle Dataset:

https://www.kaggle.com/datasets/uciml/iris

The Iris dataset from Kaggle was used as the dataset for this laboratory experiment.

---

## Files

- `Iris.csv` - Iris dataset used in this experiment
- `main.py` - Main Python program for SVM classification
- `requirements.txt` - Required Python libraries
- `README.md` - Laboratory documentation
- `images/` - Generated graphs from the experiment

---

**LAB05 - Support Vector Machine (SVM)**  
Machine Learning Laboratory
