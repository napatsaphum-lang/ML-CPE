import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("GRACE_GRACE-FO_Months_RL06.csv")

print("\n===== Dataset Preview =====")
print(df.head())

print("\n===== Shape =====")
print(df.shape)

print("\n===== Data Types =====")
print(df.dtypes)

print("\n===== Summary Statistics =====")
print(df.describe(include="all"))

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Duplicate Records =====")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n===== Class Distribution =====")

categorical_columns = df.select_dtypes(include=["object"]).columns

if len(categorical_columns) > 0:
    for column in categorical_columns:
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False))
else:
    print("No categorical columns found.")

numeric_columns = df.select_dtypes(include=np.number).columns

if len(numeric_columns) > 0:
    df[numeric_columns].hist(
        figsize=(12, 8),
        bins=20
    )

    plt.suptitle("Histogram of Numerical Features")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo numerical columns found for Histogram.")

if len(numeric_columns) > 1:
    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        correlation,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("\nNot enough numerical columns for Correlation Heatmap.")

print("\n===== Data Cleaning =====")

clean_df = df.copy()

print("\nMissing Values Before Cleaning:")
print(clean_df.isnull().sum())

for column in clean_df.columns:

    missing_count = clean_df[column].isnull().sum()

    if missing_count > 0:

        if pd.api.types.is_numeric_dtype(clean_df[column]):

            mean_value = clean_df[column].mean()
            median_value = clean_df[column].median()

            print(f"\nColumn: {column}")
            print("Missing:", missing_count)
            print("Mean:", mean_value)
            print("Median:", median_value)

            clean_df[column] = clean_df[column].fillna(
                median_value
            )

        else:

            mode_value = clean_df[column].mode()

            if not mode_value.empty:
                clean_df[column] = clean_df[column].fillna(
                    mode_value[0]
                )

print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

print("\n===== Duplicate Removal =====")

print("Duplicate Rows Before Removal:")
print(clean_df.duplicated().sum())

clean_df = clean_df.drop_duplicates()

print("Duplicate Rows After Removal:")
print(clean_df.duplicated().sum())

print("\n===== Incorrect Data Correction =====")

text_columns = clean_df.select_dtypes(
    include=["object"]
).columns

for column in text_columns:

    clean_df[column] = (
        clean_df[column]
        .astype(str)
        .str.strip()
    )

print("Text data correction completed.")

print("\n===== Data Type Conversion =====")

print("\nData Types Before Conversion:")
print(clean_df.dtypes)

for column in clean_df.columns:

    column_name = column.lower()

    if (
        "date" in column_name
        or "time" in column_name
    ):

        converted_column = pd.to_datetime(
            clean_df[column],
            errors="coerce"
        )

        if converted_column.notna().sum() > 0:
            clean_df[column] = converted_column

print("\nData Types After Conversion:")
print(clean_df.dtypes)

print("\n===== Mean vs Median =====")

numeric_columns_clean = clean_df.select_dtypes(
    include=np.number
).columns

if len(numeric_columns_clean) > 0:

    for column in numeric_columns_clean:

        mean_value = clean_df[column].mean()
        median_value = clean_df[column].median()

        print(f"\nColumn: {column}")
        print("Mean   :", mean_value)
        print("Median :", median_value)

else:
    print("No numerical columns found.")

print("\n===== Feature Engineering =====")

feature_df = clean_df.copy()

categorical_columns = feature_df.select_dtypes(
    include=["object", "category"]
).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\n===== Label Encoding =====")

label_encoded_df = feature_df.copy()

label_encoder = LabelEncoder()

if len(categorical_columns) > 0:

    for column in categorical_columns:

        label_encoded_df[column] = (
            label_encoder.fit_transform(
                label_encoded_df[column].astype(str)
            )
        )

    print(label_encoded_df.head())

else:
    print("No categorical columns available for Label Encoding.")

print("\n===== One-Hot Encoding =====")

if len(categorical_columns) > 0:

    one_hot_df = pd.get_dummies(
        feature_df,
        columns=categorical_columns,
        drop_first=False
    )

else:

    one_hot_df = feature_df.copy()

print(one_hot_df.head())

print("\nShape Before Encoding:")
print(feature_df.shape)

print("\nShape After Label Encoding:")
print(label_encoded_df.shape)

print("\nShape After One-Hot Encoding:")
print(one_hot_df.shape)

clean_df.to_csv(
    "GRACE_Cleaned_Data.csv",
    index=False
)

label_encoded_df.to_csv(
    "GRACE_Label_Encoded.csv",
    index=False
)

one_hot_df.to_csv(
    "GRACE_OneHot_Encoded.csv",
    index=False
)

print("\n===== Completed =====")

print("Files saved:")
print("1. GRACE_Cleaned_Data.csv")
print("2. GRACE_Label_Encoded.csv")
print("3. GRACE_OneHot_Encoded.csv")