import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# กำหนดตำแหน่งไฟล์และค่าที่ใช้ในโปรแกรม
# ==========================================

folder_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    folder_path,
    "iris.csv"
)

image_folder = os.path.join(
    folder_path,
    "images"
)

random_state = 42
test_size = 0.20
k_values = [3, 5, 7]


# สร้างโฟลเดอร์ images ถ้ายังไม่มี
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


# ==========================================
# โหลดข้อมูลจากไฟล์ CSV
# ==========================================

def load_dataset():
    print("กำลังโหลด Iris Dataset...")

    if not os.path.exists(data_path):
        print("ไม่พบไฟล์ iris.csv")
        print("กรุณานำไฟล์ iris.csv มาไว้ในโฟลเดอร์เดียวกับ LAB4.py")
        return None

    try:
        data = pd.read_csv(data_path)

    except pd.errors.EmptyDataError:
        print("ไฟล์ iris.csv ไม่มีข้อมูล")
        return None

    except Exception as error:
        print("ไม่สามารถอ่านไฟล์ iris.csv ได้")
        print("รายละเอียด:", error)
        return None

    required_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    for column in required_columns:
        if column not in data.columns:
            print("ไม่พบคอลัมน์:", column)
            return None

    print("โหลดข้อมูลสำเร็จ")
    print("จำนวนข้อมูลทั้งหมด:", len(data))
    print("จำนวนคอลัมน์:", len(data.columns))

    return data


# ==========================================
# สำรวจข้อมูล
# ==========================================

def explore_dataset(data):
    print("\n========================================")
    print("1. การสำรวจข้อมูล")
    print("========================================")

    print("\nตัวอย่างข้อมูล 5 แถวแรก")
    print(data.head())

    print("\nชื่อคอลัมน์")
    print(data.columns.tolist())

    print("\nขนาดของ Dataset")
    print(data.shape)

    print("\nชนิดข้อมูลของแต่ละคอลัมน์")
    print(data.dtypes)

    print("\nค่าสถิติพื้นฐาน")
    print(data.describe())

    print("\nตรวจสอบ Missing Values")
    print(data.isnull().sum())

    print("\nจำนวนข้อมูลในแต่ละ Class")
    print(data["species"].value_counts())


# ==========================================
# ทำความสะอาดข้อมูล
# ==========================================

def clean_dataset(data):
    print("\n========================================")
    print("2. การทำความสะอาดข้อมูล")
    print("========================================")

    before_cleaning = len(data)

    data = data.dropna()

    data = data.drop_duplicates()

    after_cleaning = len(data)

    print("จำนวนข้อมูลก่อนทำความสะอาด:", before_cleaning)
    print("จำนวนข้อมูลหลังทำความสะอาด:", after_cleaning)
    print(
        "จำนวนข้อมูลที่ถูกลบ:",
        before_cleaning - after_cleaning
    )

    return data.reset_index(drop=True)


# ==========================================
# แสดงกราฟ Dataset
# ==========================================

def plot_dataset(data):
    print("\nกำลังสร้างกราฟแสดงข้อมูล Iris...")

    plt.figure(figsize=(8, 6))

    species_list = data["species"].unique()

    for species in species_list:
        species_data = data[
            data["species"] == species
        ]

        plt.scatter(
            species_data["sepal_length"],
            species_data["petal_length"],
            label=species,
            alpha=0.70
        )

    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.title("Iris Dataset")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        image_folder,
        "iris_dataset.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("บันทึกกราฟ:", output_path)


# ==========================================
# ทดลอง Clustering ด้วย K-Means
# ==========================================

def clustering_example(data):
    print("\n========================================")
    print("3. Clustering ด้วย K-Means")
    print("========================================")

    cluster_features = data[
        [
            "sepal_length",
            "petal_length"
        ]
    ]

    scaler = StandardScaler()

    cluster_scaled = scaler.fit_transform(
        cluster_features
    )

    kmeans_model = KMeans(
        n_clusters=3,
        random_state=random_state,
        n_init=10
    )

    cluster_result = kmeans_model.fit_predict(
        cluster_scaled
    )

    cluster_data = data.copy()
    cluster_data["cluster"] = cluster_result

    print("\nตัวอย่างผลลัพธ์การจัดกลุ่ม")

    print(
        cluster_data[
            [
                "sepal_length",
                "petal_length",
                "species",
                "cluster"
            ]
        ].head(10)
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        cluster_scaled[:, 0],
        cluster_scaled[:, 1],
        c=cluster_result,
        alpha=0.70
    )

    plt.scatter(
        kmeans_model.cluster_centers_[:, 0],
        kmeans_model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.xlabel("Standardized Sepal Length")
    plt.ylabel("Standardized Petal Length")
    plt.title("K-Means Clustering")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        image_folder,
        "kmeans_clustering.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("บันทึกกราฟ:", output_path)


# ==========================================
# เตรียมข้อมูลสำหรับ KNN
# ==========================================

def prepare_data(data):
    print("\n========================================")
    print("4. การเตรียมข้อมูลสำหรับ KNN")
    print("========================================")

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    X = data[features]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("จำนวน Training Data:", len(X_train))
    print("จำนวน Testing Data:", len(X_test))

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("Standardize ข้อมูลเรียบร้อยแล้ว")

    print("\nค่าเฉลี่ยของ Training Data หลัง Standardize")

    print(
        np.round(
            X_train_scaled.mean(axis=0),
            4
        )
    )

    print("\nส่วนเบี่ยงเบนมาตรฐานของ Training Data")

    print(
        np.round(
            X_train_scaled.std(axis=0),
            4
        )
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


# ==========================================
# สร้างโมเดล KNN ด้วยค่า k ต่าง ๆ
# ==========================================

def train_knn_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\n========================================")
    print("5. K-Nearest Neighbors Classification")
    print("========================================")

    results = []
    models = {}

    for k in k_values:
        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        results.append(
            {
                "k": k,
                "accuracy": accuracy
            }
        )

        models[k] = model

        print(
            "k =",
            k,
            "| Accuracy =",
            round(accuracy, 4)
        )

    result_table = pd.DataFrame(results)

    print("\nตารางเปรียบเทียบ Accuracy")
    print(result_table.to_string(index=False))

    return result_table, models


# ==========================================
# หา Best K
# ==========================================

def find_best_k(result_table):
    best_index = result_table[
        "accuracy"
    ].idxmax()

    best_row = result_table.loc[
        best_index
    ]

    best_k = int(
        best_row["k"]
    )

    best_accuracy = float(
        best_row["accuracy"]
    )

    print("\n========================================")
    print("6. ผลลัพธ์ที่ดีที่สุด")
    print("========================================")

    print("Best K:", best_k)
    print(
        "Best Accuracy:",
        round(best_accuracy, 4)
    )

    return best_k, best_accuracy


# ==========================================
# สร้างกราฟเปรียบเทียบ Accuracy
# ==========================================

def plot_accuracy(result_table):
    print("\nกำลังสร้างกราฟเปรียบเทียบ Accuracy...")

    plt.figure(figsize=(8, 5))

    plt.plot(
        result_table["k"],
        result_table["accuracy"],
        marker="o"
    )

    for _, row in result_table.iterrows():
        plt.text(
            row["k"],
            row["accuracy"] + 0.002,
            str(round(row["accuracy"], 4)),
            ha="center"
        )

    plt.xticks(k_values)
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy Comparison")
    plt.ylim(
        max(0, result_table["accuracy"].min() - 0.05),
        1.02
    )
    plt.tight_layout()

    output_path = os.path.join(
        image_folder,
        "knn_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("บันทึกกราฟ:", output_path)


# ==========================================
# ประเมิน Best Model
# ==========================================

def evaluate_best_model(
    best_k,
    models,
    X_test,
    y_test
):
    print("\n========================================")
    print("7. การประเมิน Best Model")
    print("========================================")

    best_model = models[best_k]

    prediction = best_model.predict(
        X_test
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            prediction,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        prediction
    )

    class_names = sorted(
        y_test.unique()
    )

    print("Confusion Matrix")
    print(matrix)

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=20
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.title(
        "Confusion Matrix\n"
        + "Best K = "
        + str(best_k)
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_path = os.path.join(
        image_folder,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("บันทึกกราฟ:", output_path)


# ==========================================
# สรุปผลการทดลอง
# ==========================================

def discussion(
    result_table,
    best_k,
    best_accuracy
):
    print("\n========================================")
    print("8. Discussion")
    print("========================================")

    print(
        "จากการทดลองโมเดล KNN ด้วยค่า k เท่ากับ"
    )

    for _, row in result_table.iterrows():
        print(
            "k =",
            int(row["k"]),
            "ให้ Accuracy เท่ากับ",
            round(row["accuracy"], 4)
        )

    print(
        "\nค่า k ที่ให้ผลลัพธ์ดีที่สุดคือ",
        best_k
    )

    print(
        "โดยมี Accuracy เท่ากับ",
        round(best_accuracy, 4)
    )

    print(
        "\nการเลือกค่า k มีผลต่อประสิทธิภาพของโมเดล"
    )

    print(
        "หากค่า k มีค่าน้อยเกินไป โมเดลอาจไวต่อ"
        "ข้อมูลผิดปกติหรือ Noise"
    )

    print(
        "หากค่า k มีค่ามากเกินไป โมเดลอาจพิจารณา"
        "เพื่อนบ้านจำนวนมากเกินไปและทำให้แยก Class"
        "ได้ไม่ชัดเจน"
    )

    print(
        "การ Standardize ข้อมูลช่วยให้ Feature ทุกตัว"
        "อยู่ใน Scale ใกล้เคียงกัน ซึ่งมีความสำคัญกับ KNN"
        "เนื่องจาก KNN ใช้ระยะทางในการจำแนกข้อมูล"
    )


# ==========================================
# ส่วนหลักของโปรแกรม
# ==========================================

def main():
    data = load_dataset()

    if data is None:
        return

    explore_dataset(data)

    data = clean_dataset(data)

    plot_dataset(data)

    clustering_example(data)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(data)

    result_table, models = train_knn_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_k, best_accuracy = find_best_k(
        result_table
    )

    plot_accuracy(
        result_table
    )

    evaluate_best_model(
        best_k,
        models,
        X_test,
        y_test
    )

    discussion(
        result_table,
        best_k,
        best_accuracy
    )

    print("\n========================================")
    print("โปรแกรมทำงานเสร็จเรียบร้อย")
    print("========================================")

    print("\nไฟล์กราฟถูกบันทึกในโฟลเดอร์ images")

    print("1. images/iris_dataset.png")
    print("2. images/kmeans_clustering.png")
    print("3. images/knn_accuracy.png")
    print("4. images/confusion_matrix.png")


if __name__ == "__main__":
    main()