import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================
# LOAD DATASET
# ======================================

df = pd.read_csv("india_housing_prices.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ======================================
# KEEP ONLY REQUIRED COLUMNS
# ======================================

df = df[
    [
        "State",
        "City",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
    ]
]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values

df["State"].fillna(df["State"].mode()[0], inplace=True)
df["City"].fillna(df["City"].mode()[0], inplace=True)
df["Property_Type"].fillna(df["Property_Type"].mode()[0], inplace=True)

df["BHK"].fillna(df["BHK"].median(), inplace=True)
df["Size_in_SqFt"].fillna(df["Size_in_SqFt"].median(), inplace=True)

# Save median size for optional input
median_size = df["Size_in_SqFt"].median()
joblib.dump(median_size, "median_size.pkl")

# ======================================
# VISUALIZATION
# ======================================

plt.figure(figsize=(7,5))
plt.scatter(df["Size_in_SqFt"], df["Price_in_Lakhs"], alpha=0.4)
plt.xlabel("Size (SqFt)")
plt.ylabel("Price (Lakhs)")
plt.title("House Size vs Price")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(
    df[["BHK", "Size_in_SqFt", "Price_in_Lakhs"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.show()

# ======================================
# FEATURES
# ======================================

X = df.drop("Price_in_Lakhs", axis=1)
y = df["Price_in_Lakhs"]

categorical_features = ["State", "City", "Property_Type"]
numeric_features = ["BHK", "Size_in_SqFt"]

# ======================================
# PREPROCESSING
# ======================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ======================================
# MODEL
# ======================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# ======================================
# EVALUATION
# ======================================

pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("MAE :", mean_absolute_error(y_test, pred))
print("MSE :", mean_squared_error(y_test, pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, pred)))
print("R² :", r2_score(y_test, pred))

# ======================================
# SAVE MODEL
# ======================================

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")

# ======================================
# SAMPLE PREDICTION
# ======================================

sample = pd.DataFrame({
    "State": ["Tamil Nadu"],
    "City": ["Chennai"],
    "Property_Type": ["Apartment"],
    "BHK": [3],
    "Size_in_SqFt": [1500]
})

prediction = model.predict(sample)

print("\nPredicted Price :", round(prediction[0], 2), "Lakhs")