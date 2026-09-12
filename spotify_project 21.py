import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

y = df["track_popularity"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


print("\nColumns containing popularity:")

for column in df.columns:
    if "pop" in column.lower():
        print(repr(column))

print("\nFeatures:")
print(X.columns.tolist())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Target:")
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully")

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_lr = LinearRegression()

model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model_rf.fit(X_train_scaled, y_train)

y_pred_rf = model_rf.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression Results:")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)

print("\nModel Comparison:")

print("Linear Regression R2 Score:", r2_lr)
print("Random Forest R2 Score:", r2_rf)

if r2_rf > r2_lr:
    print("\nBest Model: Random Forest Regression")
else:
    print("\nBest Model: Linear Regression")


joblib.dump(model_rf, "spotify_random_forest_model.pkl")
joblib.dump(scaler, "spotify_scaler.pkl")

print("\nBest Model Saved Successfully")
print("Model: spotify_random_forest_model.pkl")
print("Scaler: spotify_scaler.pkl")

import numpy as np

new_song = np.array([[
    0.05,    # speechiness
    0.30,    # acousticness
    0.00,    # instrumentalness
    0.10,    # liveness
    0.60,    # valence
    120.0,   # tempo
    210000,  # duration_ms
    0.75,    # energy
    5,       # key
    -5.0     # loudness
]])

new_song_scaled = scaler.transform(new_song)

predicted_popularity = model_rf.predict(new_song_scaled)

print("\nNew Song Popularity Prediction:")
print("Predicted Track Popularity:", predicted_popularity[0])

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred_rf)

plt.xlabel("Actual Track Popularity")
plt.ylabel("Predicted Track Popularity")
plt.title("Actual vs Predicted Spotify Track Popularity")

plt.show()

metrics = ["MAE", "MSE", "R2 Score"]
values = [mae_rf, mse_rf, r2_rf]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values)

plt.title("Random Forest Model Performance")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Values")

plt.show()

results = pd.DataFrame({
    "Actual Popularity": y_test.values,
    "Predicted Popularity": y_pred_rf
})

print("\nPrediction Results:")
print(results.head(10))

print("\nEnter Song Details for Prediction:")

speechiness = float(input("Speechiness: "))
acousticness = float(input("Acousticness: "))
instrumentalness = float(input("Instrumentalness: "))
liveness = float(input("Liveness: "))
valence = float(input("Valence: "))
tempo = float(input("Tempo: "))
duration_ms = float(input("Duration (ms): "))
energy = float(input("Energy: "))
key = float(input("Key: "))
loudness = float(input("Loudness: "))

new_song = [[
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    duration_ms,
    energy,
    key,
    loudness
]]

new_song_scaled = scaler.transform(new_song)

prediction = model_rf.predict(new_song_scaled)

print("\nPredicted Spotify Track Popularity:", prediction[0])