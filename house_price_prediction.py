# House Price Prediction using Linear Regression
# Dataset: California Housing (from sklearn)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target * 100000

print("Dataset Shape:", df.shape)
print(df.head())
print(df.describe())

# Check missing values
print("Missing values:", df.isnull().sum())

# --- Chart 1: Price Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(df['Price'], bins=50, color='steelblue', edgecolor='white')
plt.axvline(df['Price'].mean(), color='red', linestyle='--', label='Mean Price')
plt.title('House Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('1_price_distribution.png')
plt.show()

# --- Chart 2: Correlation Heatmap ---
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('2_correlation.png')
plt.show()

# --- Chart 3: Income vs Price ---
plt.figure(figsize=(7, 5))
plt.scatter(df['MedInc'], df['Price'], alpha=0.3, color='teal', s=8)
plt.title('Median Income vs House Price')
plt.xlabel('Median Income')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.savefig('3_income_vs_price.png')
plt.show()

# Prepare data for model
X = df.drop('Price', axis=1)
y = df['Price']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features so all are on same range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Check model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Results:")
print(f"MAE      : ${mae:,.2f}")
print(f"RMSE     : ${rmse:,.2f}")
print(f"R2 Score : {r2:.4f}")

# --- Chart 4: Actual vs Predicted ---
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted Price (R2 = {r2:.2f})')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.tight_layout()
plt.savefig('4_actual_vs_predicted.png')
plt.show()

# --- Chart 5: Feature Importance ---
feature_names = df.drop('Price', axis=1).columns
coefficients = np.abs(model.coef_)

plt.figure(figsize=(9, 5))
plt.bar(feature_names, coefficients, color='steelblue', edgecolor='white')
plt.title('Feature Importance (Model Coefficients)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('5_feature_importance.png')
plt.show()

print("\nDone! All charts saved.")