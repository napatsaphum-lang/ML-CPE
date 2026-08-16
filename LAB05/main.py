import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


os.makedirs("images", exist_ok=True)


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


X = data[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

y = data["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== Train Test Split =====")
print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


linear_model = SVC(kernel="linear")
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
linear_acc = accuracy_score(y_test, linear_pred)


poly_model = SVC(kernel="poly")
poly_model.fit(X_train, y_train)

poly_pred = poly_model.predict(X_test)
poly_acc = accuracy_score(y_test, poly_pred)


rbf_model = SVC(kernel="rbf")
rbf_model.fit(X_train, y_train)

rbf_pred = rbf_model.predict(X_test)
rbf_acc = accuracy_score(y_test, rbf_pred)


print("\n===== SVM Accuracy =====")
print("Linear Kernel     :", round(linear_acc * 100, 2), "%")
print("Polynomial Kernel :", round(poly_acc * 100, 2), "%")
print("RBF Kernel        :", round(rbf_acc * 100, 2), "%")


accuracy_result = {
    "Linear": linear_acc,
    "Polynomial": poly_acc,
    "RBF": rbf_acc
}

best_kernel = max(
    accuracy_result,
    key=accuracy_result.get
)

print("\n===== Best Kernel =====")
print("Best Kernel:", best_kernel)
print(
    "Accuracy:",
    round(accuracy_result[best_kernel] * 100, 2),
    "%"
)


result = pd.DataFrame({
    "Actual": y_test.values,
    "Linear": linear_pred,
    "Polynomial": poly_pred,
    "RBF": rbf_pred
})

print("\n===== Prediction Results =====")
print(result.to_string(index=False))


kernels = [
    "Linear",
    "Polynomial",
    "RBF"
]

accuracy = [
    linear_acc * 100,
    poly_acc * 100,
    rbf_acc * 100
]


plt.figure(figsize=(7, 5))

bars = plt.bar(
    kernels,
    accuracy
)

plt.title("SVM Kernel Accuracy")
plt.xlabel("Kernel")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        str(round(value, 2)) + "%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "images/svm_accuracy.png",
    dpi=300
)
plt.show()


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
plt.savefig(
    "images/species_count.png",
    dpi=300
)
plt.show()


plt.figure(figsize=(7, 5))

for species in data["Species"].unique():

    iris = data[
        data["Species"] == species
    ]

    plt.scatter(
        iris["PetalLengthCm"],
        iris["PetalWidthCm"],
        label=species
    )

plt.title("Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.tight_layout()
plt.savefig(
    "images/iris_scatter.png",
    dpi=300
)
plt.show()


print("\n===== Images Saved =====")
print("images/svm_accuracy.png")
print("images/species_count.png")
print("images/iris_scatter.png")