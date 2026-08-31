from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_data(data):

    # ลบ Id เพราะไม่ใช้ในการจำแนก
    if "Id" in data.columns:
        data = data.drop("Id", axis=1)

    # เลือก Feature
    X = data[
        [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    ]

    # Target
    y = data["Species"]

    # แปลงชื่อ Species เป็นตัวเลข
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("\nClasses:")
    print(encoder.classes_)

    return X, y, encoder


def standardize_data(X_train, X_val, X_test):

    scaler = StandardScaler()

    # เรียนรู้ค่าเฉลี่ยและส่วนเบี่ยงเบนจาก Train
    X_train = scaler.fit_transform(X_train)

    # ใช้ค่าเดิมกับ Validation และ Test
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler