import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # ใช้แบ่งข้อมูล Train และ Test
from sklearn.preprocessing import StandardScaler      # ใช้ปรับ Scale ของข้อมูล
from sklearn.svm import SVC                           # ใช้สร้างโมเดล SVM
from sklearn.metrics import accuracy_score            # ใช้คำนวณ Accuracy

os.makedirs("images", exist_ok=True)  # สร้างโฟลเดอร์ images สำหรับเก็บกราฟ

# ===== 1. Load Dataset =====
data = pd.read_csv("Iris.csv")  # อ่านข้อมูล Iris Dataset จากไฟล์ CSV

print("===== Iris Dataset =====")
print(data.head())  # แสดงข้อมูล 5 แถวแรก

print("\n===== Dataset Information =====")
print("Rows:", data.shape[0])        # แสดงจำนวนแถว
print("Columns:", data.shape[1])     # แสดงจำนวนคอลัมน์

print("\n===== Missing Values =====")
print(data.isnull().sum())           # ตรวจสอบจำนวนข้อมูลที่หายไป

print("\n===== Species =====")
print(data["Species"].value_counts())  # นับจำนวนข้อมูลของแต่ละ Species


# ===== 2. Features and Target =====
features = [
    "SepalLengthCm",   # ความยาวกลีบเลี้ยง
    "SepalWidthCm",    # ความกว้างกลีบเลี้ยง
    "PetalLengthCm",   # ความยาวกลีบดอก
    "PetalWidthCm"     # ความกว้างกลีบดอก
]

X = data[features]       # กำหนด Features ที่ใช้เป็น Input ของโมเดล
y = data["Species"]      # กำหนด Species เป็น Target ที่ต้องการทำนาย


# ===== 3. Train / Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                # ข้อมูล Features และ Target
    test_size=0.2,       # แบ่ง Test 20% และ Train 80%
    random_state=42      # กำหนดการสุ่มให้ได้ข้อมูลชุดเดิมทุกครั้ง
)

print("\n===== Train Test Split =====")
print("Training Data:", len(X_train))  # จำนวนข้อมูลสำหรับ Train
print("Testing Data:", len(X_test))     # จำนวนข้อมูลสำหรับ Test


# ===== 4. Standardization =====
scaler = StandardScaler()  # สร้างตัวปรับ Scale ของ Features

X_train = scaler.fit_transform(X_train)  # เรียนรู้ Scale จาก Train และปรับข้อมูล
X_test = scaler.transform(X_test)        # ปรับ Test ด้วย Scale ที่ได้จาก Train


# ===== 5. Train SVM Models =====
kernels = {
    "Linear": "linear",       # Linear Kernel
    "Polynomial": "poly",     # Polynomial Kernel
    "RBF": "rbf"              # Radial Basis Function Kernel
}

predictions = {}      # เก็บผลการทำนายของแต่ละ Kernel
accuracy_result = {}  # เก็บค่า Accuracy ของแต่ละ Kernel

for name, kernel in kernels.items():       # วนทดสอบ SVM ทั้ง 3 Kernel
    model = SVC(kernel=kernel)             # สร้าง SVM ตาม Kernel ปัจจุบัน
    model.fit(X_train, y_train)            # Train โมเดลด้วย Training Data

    pred = model.predict(X_test)           # ทำนาย Species จาก Testing Data
    predictions[name] = pred               # เก็บผล Prediction
    accuracy_result[name] = accuracy_score(y_test, pred)  # คำนวณ Accuracy


# ===== 6. Show Accuracy =====
print("\n===== SVM Accuracy =====")

for name, acc in accuracy_result.items():  # วนแสดง Accuracy ของแต่ละ Kernel
    print(f"{name} Kernel: {acc * 100:.2f}%")


# ===== 7. Find Best Kernel =====
best_kernel = max(
    accuracy_result,
    key=accuracy_result.get  # เลือก Kernel ที่มี Accuracy สูงที่สุด
)

print("\n===== Best Kernel =====")
print("Best Kernel:", best_kernel)  # แสดง Kernel ที่ดีที่สุด
print(f"Accuracy: {accuracy_result[best_kernel] * 100:.2f}%")  # แสดง Accuracy


# ===== 8. Prediction Results =====
result = pd.DataFrame({
    "Actual": y_test.values,  # คำตอบจริง
    **predictions             # ผลทำนายของ Linear, Polynomial และ RBF
})

print("\n===== Prediction Results =====")
print(result.to_string(index=False))  # แสดงผลจริงเทียบกับผล Prediction


# ===== 9. Graph: SVM Accuracy =====
names = list(accuracy_result.keys())                 # ชื่อ Kernel
scores = [acc * 100 for acc in accuracy_result.values()]  # Accuracy เป็น %

plt.figure(figsize=(7, 5))            # กำหนดขนาดกราฟ
bars = plt.bar(names, scores)          # สร้างกราฟแท่ง Accuracy

plt.title("SVM Kernel Accuracy")       # ชื่อกราฟ
plt.xlabel("Kernel")                   # ชื่อแกน X
plt.ylabel("Accuracy (%)")             # ชื่อแกน Y
plt.ylim(0, 105)                       # กำหนดช่วงแกน Y

for bar, value in zip(bars, scores):   # แสดงค่า Accuracy บนแต่ละแท่ง
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()                              # จัดตำแหน่งกราฟให้พอดี
plt.savefig("images/svm_accuracy.png", dpi=300) # บันทึกกราฟเป็นไฟล์ PNG
plt.show()                                      # แสดงกราฟ


# ===== 10. Graph: Species Count =====
species_count = data["Species"].value_counts()  # นับจำนวนแต่ละ Species

plt.figure(figsize=(7, 5))
bars = plt.bar(species_count.index, species_count.values)  # สร้างกราฟจำนวน Species

plt.title("Number of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")

for bar, value in zip(bars, species_count.values):  # แสดงจำนวนบนแท่งกราฟ
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.savefig("images/species_count.png", dpi=300)  # บันทึกกราฟจำนวน Species
plt.show()


# ===== 11. Graph: Iris Scatter Plot =====
plt.figure(figsize=(7, 5))

for species, group in data.groupby("Species"):  # แยกข้อมูลตาม Species
    plt.scatter(
        group["PetalLengthCm"],  # แกน X = Petal Length
        group["PetalWidthCm"],   # แกน Y = Petal Width
        label=species            # ชื่อ Species
    )

plt.title("Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()  # แสดงชื่อ Species ในกราฟ

plt.tight_layout()
plt.savefig("images/iris_scatter.png", dpi=300)  # บันทึก Scatter Plot
plt.show()


# ===== 12. Show Saved Files =====
print("\n===== Images Saved =====")

for file in [
    "svm_accuracy.png",
    "species_count.png",
    "iris_scatter.png"
]:
    print("images/" + file)  # แสดงตำแหน่งไฟล์รูปที่บันทึก
