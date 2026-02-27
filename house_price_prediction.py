"""
╔══════════════════════════════════════════════════════╗
║         HOUSE PRICE PREDICTION                       ║
║   Python | Pandas | Sklearn | Matplotlib | Seaborn   ║
╚══════════════════════════════════════════════════════╝

Dataset: Boston Housing / California Housing
Auto-loads from sklearn datasets if no CSV found
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F']

# ============================================================
#  LOAD DATA
# ============================================================
def load_data():
    print("\n📂 Loading Housing Dataset...")
    try:
        df = pd.read_csv('housing.csv')
        print(f"✅ Dataset loaded from CSV: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠️  No CSV found — loading California Housing dataset from sklearn...")
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['Price'] = housing.target * 100000  # Convert to dollars
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

# ============================================================
#  DATA EXPLORATION
# ============================================================
def explore_data(df):
    print("\n📊 Dataset Info:")
    print(f"  Shape        : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Missing Values: {df.isnull().sum().sum()}")
    print(f"\n  Basic Stats:")
    print(df.describe().round(2))

# ============================================================
#  ANALYSIS 1: Price Distribution
# ============================================================
def analysis_price_distribution(df):
    print("\n📊 Analysis 1: Price Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['Price'], bins=50, color='#FF6B6B', edgecolor='white', alpha=0.8)
    axes[0].axvline(df['Price'].mean(), color='#4ECDC4', linestyle='--',
                    linewidth=2, label=f'Mean: ${df["Price"].mean():,.0f}')
    axes[0].axvline(df['Price'].median(), color='#F7DC6F', linestyle='--',
                    linewidth=2, label=f'Median: ${df["Price"].median():,.0f}')
    axes[0].set_title('House Price Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(df['Price'], patch_artist=True,
                    boxprops=dict(facecolor='#FF6B6B', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2))
    axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price ($)')

    plt.suptitle('🏠 House Price Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_price_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_price_distribution.png")

# ============================================================
#  ANALYSIS 2: Correlation Heatmap
# ============================================================
def analysis_correlation(df):
    print("\n📊 Analysis 2: Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, mask=mask, linewidths=0.5,
                annot_kws={'size': 9})
    ax.set_title('🔥 Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('2_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_correlation.png")

# ============================================================
#  ANALYSIS 3: Feature vs Price
# ============================================================
def analysis_feature_vs_price(df):
    print("\n📊 Analysis 3: Top Features vs Price...")
    corr_with_price = df.corr()['Price'].abs().sort_values(ascending=False)
    top_features    = corr_with_price[1:5].index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        axes[i].scatter(df[feature], df['Price'],
                        alpha=0.3, color=COLORS[i], s=10)
        # Trend line
        z = np.polyfit(df[feature], df['Price'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        axes[i].plot(x_line, p(x_line), color='white', linewidth=2, linestyle='--')
        axes[i].set_xlabel(feature, fontsize=11)
        axes[i].set_ylabel('Price ($)', fontsize=11)
        axes[i].set_title(f'{feature} vs Price', fontsize=12, fontweight='bold')

    plt.suptitle('📈 Top Features vs House Price', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_feature_vs_price.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 3_feature_vs_price.png")

# ============================================================
#  ML: LINEAR REGRESSION
# ============================================================
def train_model(df):
    print("\n🤖 Training Linear Regression Model...")

    # Features & Target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n✅ Model Performance:")
    print(f"   MAE  : ${mae:,.2f}")
    print(f"   RMSE : ${rmse:,.2f}")
    print(f"   R²   : {r2:.4f} ({r2*100:.2f}% variance explained)")

    return model, scaler, X_test, y_test, y_pred, r2, mae

# ============================================================
#  ANALYSIS 4: Actual vs Predicted
# ============================================================
def analysis_actual_vs_predicted(y_test, y_pred, r2):
    print("\n📊 Analysis 4: Actual vs Predicted...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.3, color='#4ECDC4', s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                 color='#FF6B6B', linewidth=2, linestyle='--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=11)
    axes[0].set_title(f'Actual vs Predicted (R² = {r2:.4f})',
                      fontsize=13, fontweight='bold')
    axes[0].legend()

    # Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=50, color='#45B7D1', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='#FF6B6B', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Residuals ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residuals Distribution', fontsize=13, fontweight='bold')

    plt.suptitle('🤖 Linear Regression Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_actual_vs_predicted.png")

# ============================================================
#  ANALYSIS 5: Feature Importance
# ============================================================
def analysis_feature_importance(model, df):
    print("\n📊 Analysis 5: Feature Importance...")
    features    = df.drop('Price', axis=1).columns
    importances = np.abs(model.coef_)
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(features)),
                  importances[indices],
                  color=COLORS[:len(features)],
                  edgecolor='white')

    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.2f}', ha='center',
                fontweight='bold', fontsize=9)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=30, ha='right')
    ax.set_title('📊 Feature Importance (Linear Regression Coefficients)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('5_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_feature_importance.png")

# ============================================================
#  PREDICT NEW HOUSE
# ============================================================
def predict_new_house(model, scaler, df):
    print("\n🏠 Sample House Price Prediction:")
    sample = df.drop('Price', axis=1).iloc[0:1].copy()
    sample_scaled = scaler.transform(sample)
    predicted_price = model.predict(sample_scaled)[0]
    print(f"   Features     : {sample.values[0]}")
    print(f"   Predicted Price: ${predicted_price:,.2f}")

# ============================================================
#  SUMMARY
# ============================================================
def print_summary(df, r2, mae):
    print("\n" + "="*55)
    print("       🏠 HOUSE PRICE PREDICTION SUMMARY")
    print("="*55)
    print(f"  Total Houses      : {len(df):,}")
    print(f"  Features Used     : {df.shape[1]-1}")
    print(f"  Average Price     : ${df['Price'].mean():,.2f}")
    print(f"  Min Price         : ${df['Price'].min():,.2f}")
    print(f"  Max Price         : ${df['Price'].max():,.2f}")
    print(f"  Model             : Linear Regression")
    print(f"  R² Score          : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  Mean Abs Error    : ${mae:,.2f}")
    print("="*55)

# ============================================================
#  MAIN
# ============================================================
def main():
    print("="*55)
    print("       🏠 HOUSE PRICE PREDICTION")
    print("   Python | Pandas | Sklearn | Matplotlib")
    print("="*55)

    df = load_data()
    explore_data(df)

    # Analyses
    analysis_price_distribution(df)
    analysis_correlation(df)
    analysis_feature_vs_price(df)

    # ML
    model, scaler, X_test, y_test, y_pred, r2, mae = train_model(df)
    analysis_actual_vs_predicted(y_test, y_pred, r2)
    analysis_feature_importance(model, df)
    predict_new_house(model, scaler, df)

    print_summary(df, r2, mae)
    print("\n✅ All analyses complete!")
    print("📁 5 charts saved as PNG files!")

if __name__ == "__main__":
    main()