import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# สร้างโฟลเดอร์สำหรับเก็บกราฟ
os.makedirs("images", exist_ok=True)


# ===== 1. Load Dataset =====
data = pd.read_csv("Iris.csv")

print("===== Iris Dataset =====")
print(data.head())

print("\n===== Dataset Information =====")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n===== Missing Values =====")
print(data.isnull().sum())

print("\n===== Species =====")
print(data["Species"].value_counts())


# ===== 2. กำหนด Features (X) และ Target (y) =====
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = data[features]
y = data["Species"]


# ===== 3. แบ่งข้อมูล Train 80% และ Test 20% =====
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== Train Test Split =====")
print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))


# ===== 4. ปรับ Scale ของข้อมูล =====
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===== 5. สร้างและทดสอบ SVM ทั้ง 3 Kernel =====
kernels = {
    "Linear": "linear",
    "Polynomial": "poly",
    "RBF": "rbf"
}

predictions = {}
accuracy_result = {}

for name, kernel in kernels.items():
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred
    accuracy_result[name] = accuracy_score(y_test, pred)


# ===== 6. แสดง Accuracy ของแต่ละ Kernel =====
print("\n===== SVM Accuracy =====")

for name, acc in accuracy_result.items():
    print(f"{name} Kernel: {acc * 100:.2f}%")


# ===== 7. หา Kernel ที่มี Accuracy สูงที่สุด =====
best_kernel = max(
    accuracy_result,
    key=accuracy_result.get
)

print("\n===== Best Kernel =====")
print("Best Kernel:", best_kernel)
print(f"Accuracy: {accuracy_result[best_kernel] * 100:.2f}%")


# ===== 8. แสดงผลจริงเทียบกับผลการทำนาย =====
result = pd.DataFrame({
    "Actual": y_test.values,
    **predictions
})

print("\n===== Prediction Results =====")
print(result.to_string(index=False))


# ===== 9. กราฟเปรียบเทียบ Accuracy =====
names = list(accuracy_result.keys())
scores = [acc * 100 for acc in accuracy_result.values()]

plt.figure(figsize=(7, 5))
bars = plt.bar(names, scores)

plt.title("SVM Kernel Accuracy")
plt.xlabel("Kernel")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for bar, value in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("images/svm_accuracy.png", dpi=300)
plt.show()


# ===== 10. กราฟแสดงจำนวนข้อมูลแต่ละ Species =====
species_count = data["Species"].value_counts()

plt.figure(figsize=(7, 5))
bars = plt.bar(
    species_count.index,
    species_count.values
)

plt.title("Number of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")

for bar, value in zip(bars, species_count.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.savefig("images/species_count.png", dpi=300)
plt.show()


# ===== 11. Scatter Plot แสดงการกระจายของ Species =====
plt.figure(figsize=(7, 5))

for species, group in data.groupby("Species"):
    plt.scatter(
        group["PetalLengthCm"],
        group["PetalWidthCm"],
        label=species
    )

plt.title("Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.tight_layout()
plt.savefig("images/iris_scatter.png", dpi=300)
plt.show()


# ===== 12. แสดงไฟล์กราฟที่บันทึก =====
print("\n===== Images Saved =====")

for file in [
    "svm_accuracy.png",
    "species_count.png",
    "iris_scatter.png"
]:
    print("images/" + file)
