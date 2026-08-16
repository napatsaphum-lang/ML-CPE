# LAB05 - Support Vector Machine (SVM)

## Overview

This project is part of **LAB05: Support Vector Machine and SVM Applications**.

The purpose of this lab is to study and apply the Support Vector Machine (SVM) algorithm for classification. The Iris dataset is used to train and evaluate SVM models with different kernel functions.

The performance of three SVM kernels is compared:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel

---

## Dataset

This project uses the **Iris Dataset** for classification.

The dataset contains **150 samples** of Iris flowers and consists of four input features:

| Feature | Description |
|---|---|
| SepalLengthCm | Sepal length in centimeters |
| SepalWidthCm | Sepal width in centimeters |
| PetalLengthCm | Petal length in centimeters |
| PetalWidthCm | Petal width in centimeters |

The target variable is `Species`, which contains three classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## Project Structure

```text
LAB05/
│
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

## Program Process

The program follows these steps:

1. Load the Iris dataset from `Iris.csv`.
2. Display and explore the dataset.
3. Check for missing values.
4. Select the input features and target variable.
5. Split the dataset into training and testing data.
6. Standardize the input features using `StandardScaler`.
7. Train the SVM model using Linear Kernel.
8. Train the SVM model using Polynomial Kernel.
9. Train the SVM model using RBF Kernel.
10. Predict the test dataset.
11. Calculate and compare the accuracy of each kernel.
12. Display and save the result graphs.

---

## Libraries

The following Python libraries are used:

- pandas
- matplotlib
- scikit-learn

Install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Running the Program

Open the terminal in the `LAB05` directory and run:

```bash
python main.py
```

The program will display the dataset information, prediction results, accuracy scores, and graphs.

---

## Experimental Results

The SVM models were evaluated using the testing dataset.

| SVM Kernel | Accuracy |
|---|---:|
| Linear | 96.67% |
| Polynomial | 96.67% |
| RBF | 100.00% |

Based on this experiment, the **RBF Kernel achieved the highest accuracy of 100.00%** on the selected train/test split.

---

## SVM Kernel Accuracy

The following graph compares the classification accuracy of the three SVM kernels.

![SVM Kernel Accuracy](images/svm_accuracy.png)

---

## Iris Species Distribution

The following graph shows the number of samples for each Iris species in the dataset.

![Number of Iris Species](images/species_count.png)

---

## Iris Dataset Visualization

The scatter plot below shows the relationship between **Petal Length** and **Petal Width** for the three Iris species.

![Iris Dataset Scatter Plot](images/iris_scatter.png)

---

## Conclusion

In this lab, Support Vector Machine was applied to classify the Iris dataset using three different kernel functions: Linear, Polynomial, and RBF.

The input features were standardized before training, and the dataset was divided into training and testing sets. The performance of each model was evaluated using accuracy.

The experimental results showed that the **RBF Kernel achieved the highest accuracy at 100.00%**, while the Linear and Polynomial Kernels achieved **96.67%**.

This experiment demonstrates that the selection of an appropriate SVM kernel can affect classification performance.

---

## Files

- `Iris.csv` - Iris dataset used in this experiment.
- `main.py` - Main Python program for training and evaluating the SVM models.
- `requirements.txt` - Required Python libraries.
- `images/` - Generated graphs from the experiment.

---

**LAB05 - Support Vector Machine (SVM)**
