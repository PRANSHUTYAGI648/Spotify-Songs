import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset shape
print("Dataset Shape:", df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns
numeric_columns = df.select_dtypes(include=['number']).columns

print("\nNumerical Columns:")
print(numeric_columns)

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Spotify Songs - Correlation Heatmap")
plt.show()

print("\nNumerical Features:")
print(df.select_dtypes(include=['number']).columns.tolist())

print("\nNumber of Unique Values:")
print(df.nunique().sort_values())

print("\nAll Columns:")
for column in df.columns:
    print(column, "->", df[column].dtype)

print("\nFirst 5 Rows:")
print(df.head())

print("\nTarget Candidates:")
print(df.columns.tolist())

X = df[
    [
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "energy",
        "key",
        "loudness"
    ]
]

print("\nColumns containing popularity:")

for column in df.columns:
    if "pop" in column.lower():
        print(repr(column))

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)




