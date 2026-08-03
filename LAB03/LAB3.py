import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# กำหนดตำแหน่งไฟล์ข้อมูล
folder_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(folder_path, "age_gender.csv")

# กำหนดค่าที่ใช้ในโปรแกรม
sample_size = 6000
pca_components = 50
random_state = 42


# โหลดข้อมูลจากไฟล์ CSV
def load_dataset():
    print("กำลังโหลดข้อมูล...")

    if not os.path.exists(data_path):
        print("ไม่พบไฟล์ age_gender.csv")
        print("กรุณานำไฟล์มาไว้ในโฟลเดอร์เดียวกับ LAB3.py")
        return None

    data = pd.read_csv(data_path)

    columns = ["age", "gender", "pixels"]

    for column in columns:
        if column not in data.columns:
            print("ไม่พบคอลัมน์", column)
            return None

    data = data.dropna(
        subset=["age", "gender", "pixels"]
    )

    if len(data) > sample_size:
        data = data.sample(
            n=sample_size,
            random_state=random_state
        )

    data = data.reset_index(drop=True)

    print("จำนวนข้อมูลที่ใช้:", len(data))
    print(data[["age", "gender"]].head())

    return data


# แปลงข้อมูล Pixel ให้เป็นตัวเลข
def prepare_data(data):
    print("\nกำลังเตรียมข้อมูลภาพ...")

    pixel_list = []

    for pixel in data["pixels"]:
        pixel_value = np.fromstring(
            str(pixel),
            sep=" ",
            dtype=np.float32
        )

        pixel_list.append(pixel_value)

    images = np.array(pixel_list)

    # ปรับค่า Pixel ให้อยู่ระหว่าง 0 ถึง 1
    images = images / 255.0

    ages = data["age"].astype(float).values
    genders = data["gender"].astype(int).values

    print("จำนวนภาพ:", images.shape[0])
    print("จำนวน Pixel ต่อภาพ:", images.shape[1])

    return images, ages, genders


# แสดงตัวอย่างภาพใน Dataset
def show_sample_images(images, ages, genders):
    plt.figure(figsize=(10, 5))

    for i in range(5):
        plt.subplot(1, 5, i + 1)

        image = images[i].reshape(48, 48)

        if genders[i] == 0:
            gender_name = "Male"
        else:
            gender_name = "Female"

        plt.imshow(image, cmap="gray")
        plt.title(
            "Age: " + str(int(ages[i])) +
            "\n" + gender_name
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()


# Simple Linear Regression
def simple_regression(images, ages):
    print("\n----------------------------------")
    print("Simple Linear Regression")
    print("----------------------------------")

    # ใช้ค่าความสว่างเฉลี่ยของภาพเป็น Feature
    brightness = images.mean(axis=1)

    X = brightness.reshape(-1, 1)
    y = ages

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    age_prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        age_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            age_prediction
        )
    )

    r2 = r2_score(
        y_test,
        age_prediction
    )

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    sort_index = np.argsort(X_test[:, 0])

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_test,
        y_test,
        alpha=0.4,
        label="Actual Age"
    )

    plt.plot(
        X_test[sort_index],
        age_prediction[sort_index],
        label="Regression Line"
    )

    plt.xlabel("Average Brightness")
    plt.ylabel("Age")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_linear_regression.png")
    plt.show()


# Multiple Linear Regression และ PCA
def multiple_regression(images, ages):
    print("\n----------------------------------")
    print("Multiple Linear Regression")
    print("----------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        ages,
        test_size=0.20,
        random_state=random_state
    )

    # ปรับข้อมูลให้อยู่ในมาตรฐานเดียวกัน
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ลดจำนวน Feature ด้วย PCA
    pca = PCA(
        n_components=pca_components,
        random_state=random_state
    )

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    variance = (
        pca.explained_variance_ratio_.sum()
        * 100
    )

    print("จำนวน Feature เดิม:", images.shape[1])
    print("จำนวน Feature หลัง PCA:", pca_components)
    print("Explained Variance:", round(variance, 2), "%")

    model = LinearRegression()

    model.fit(
        X_train_pca,
        y_train
    )

    age_prediction = model.predict(
        X_test_pca
    )

    mae = mean_absolute_error(
        y_test,
        age_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            age_prediction
        )
    )

    r2 = r2_score(
        y_test,
        age_prediction
    )

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_test,
        age_prediction,
        alpha=0.4
    )

    minimum = min(
        y_test.min(),
        age_prediction.min()
    )

    maximum = max(
        y_test.max(),
        age_prediction.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual Age and Predicted Age")
    plt.tight_layout()
    plt.savefig("multiple_linear_regression.png")
    plt.show()

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, pca_components + 1),
        cumulative_variance
    )

    plt.xlabel("PCA Components")
    plt.ylabel("Cumulative Variance")
    plt.title("PCA Explained Variance")
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png")
    plt.show()


# Classification สำหรับจำแนกเพศ
def gender_classification(images, genders):
    print("\n----------------------------------")
    print("Gender Classification")
    print("----------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        genders,
        test_size=0.20,
        random_state=random_state,
        stratify=genders
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(
        n_components=pca_components,
        random_state=random_state
    )

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state
    )

    model.fit(
        X_train_pca,
        y_train
    )

    gender_prediction = model.predict(
        X_test_pca
    )

    gender_probability = model.predict_proba(
        X_test_pca
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        gender_prediction
    )

    precision = precision_score(
        y_test,
        gender_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        gender_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        gender_prediction,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        gender_probability
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(auc, 4))

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            gender_prediction,
            target_names=["Male", "Female"],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        gender_prediction
    )

    print("Confusion Matrix")
    print(matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.xticks(
        [0, 1],
        ["Male", "Female"]
    )

    plt.yticks(
        [0, 1],
        ["Male", "Female"]
    )

    plt.xlabel("Predicted Gender")
    plt.ylabel("Actual Gender")
    plt.title("Confusion Matrix")

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    false_positive_rate, true_positive_rate, value = roc_curve(
        y_test,
        gender_probability
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label="AUC = " + str(round(auc, 3))
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()


# ส่วนหลักของโปรแกรม
def main():
    data = load_dataset()

    if data is None:
        return

    images, ages, genders = prepare_data(data)

    show_sample_images(
        images,
        ages,
        genders
    )

    simple_regression(
        images,
        ages
    )

    multiple_regression(
        images,
        ages
    )

    gender_classification(
        images,
        genders
    )

    print("\n----------------------------------")
    print("โปรแกรมทำงานเสร็จเรียบร้อย")
    print("----------------------------------")

    print("ไฟล์กราฟที่ได้")

    print("1. sample_images.png")
    print("2. simple_linear_regression.png")
    print("3. multiple_linear_regression.png")
    print("4. pca_explained_variance.png")
    print("5. confusion_matrix.png")
    print("6. roc_curve.png")


main()