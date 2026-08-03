# LAB03 : Regression & Classification

## Machine Learning Laboratory 3

This project demonstrates the application of **Supervised Machine Learning** using Regression and Classification techniques. The project predicts a person's age and classifies gender from facial image data using Python and Scikit-learn.

---

## Objectives

- Understand the concepts of Regression and Classification.
- Apply Simple Linear Regression for age prediction.
- Apply Multiple Linear Regression with PCA.
- Apply Logistic Regression for gender classification.
- Reduce feature dimensions using Principal Component Analysis (PCA).
- Evaluate model performance using appropriate metrics.

---

## Dataset

**Dataset:** Age Gender Dataset (UTKFace)

The dataset contains:

- Age
- Gender
- Facial image pixel values (48 × 48 grayscale)

**Source**

https://www.kaggle.com/datasets/jangedoo/utkface-new

> **Note:** The dataset (`age_gender.csv`) is not included in this repository. Please download it from Kaggle and place it in the LAB03 folder before running the program.

---

## Development Tools

- Python
- Visual Studio Code
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Machine Learning Models

### Regression

- Simple Linear Regression
- Multiple Linear Regression

These models are used to predict a person's age from facial image data.

### Classification

- Logistic Regression

This model is used to classify gender into:

- Male
- Female

### Feature Reduction

- Principal Component Analysis (PCA)

PCA is used to reduce the number of image features before training the machine learning models.

---

## Evaluation Metrics

### Regression

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Area Under Curve (AUC)

---

## Project Structure

```text
LAB03
│
├── LAB3.py
├── README.md
├── requirements.txt
├── .gitignore
└── images
    ├── sample_images.png
    ├── simple_linear_regression.png
    ├── multiple_linear_regression.png
    ├── pca_explained_variance.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## Installation

Install the required libraries.

```bash
pip install -r requirements.txt
```

---

## Run the Program

```bash
python LAB3.py
```

---

## Program Workflow

1. Load the dataset (`age_gender.csv`)
2. Clean and prepare the data
3. Convert pixel values into numerical arrays
4. Split the dataset into training and testing sets
5. Train the Simple Linear Regression model
6. Apply PCA for feature reduction
7. Train the Multiple Linear Regression model
8. Train the Logistic Regression model
9. Evaluate the performance of each model
10. Display graphs and save the output images

---

## Output

The program generates the following output images:

- Sample Images
- Simple Linear Regression
- Multiple Linear Regression
- PCA Explained Variance
- Confusion Matrix
- ROC Curve

The evaluation results displayed in the terminal include:

### Regression

- MAE
- RMSE
- R² Score

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- AUC

---

## Sample Results

### Sample Images

![Sample Images](images/sample_images.png)

### Simple Linear Regression

![Simple Linear Regression](images/simple_linear_regression.png)

### Multiple Linear Regression

![Multiple Linear Regression](images/multiple_linear_regression.png)

### PCA Explained Variance

![PCA Explained Variance](images/pca_explained_variance.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

---

## Conclusion

This project demonstrates the implementation of supervised learning algorithms for regression and classification tasks using facial image data. PCA helps reduce the dimensionality of image features while maintaining useful information, improving training efficiency. The evaluation metrics and visualizations provide insight into the performance of each machine learning model.
