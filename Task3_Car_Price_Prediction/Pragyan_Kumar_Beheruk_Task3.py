"""
CAR PRICE PREDICTION WITH MACHINE LEARNING
Author: Pragyan Kumar Beheruk
Internship Task 3 - OIBSIP

This project builds machine learning models to predict car selling prices
based on various features like:
- Year of manufacture
- Present market price
- Kilometers driven
- Fuel type (Petrol, Diesel, CNG)
- Selling type (Individual, Dealer)
- Transmission (Manual, Automatic)
- Number of previous owners

Multiple regression models are trained and compared to find the best predictor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==========================================
# 1. LOAD AND EXPLORE DATA
# ==========================================

print("=" * 70)
print("CAR PRICE PREDICTION - MACHINE LEARNING PROJECT")
print("=" * 70)
print("\n[Step 1] Loading and Exploring Dataset...")

# Load data
df = pd.read_csv('car data.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Total Records: {len(df)}")
print(f"  Total Features: {len(df.columns)}")

print(f"\n[Dataset Preview]")
print(df.head(10))

print(f"\n[Dataset Information]")
print(f"  Columns: {list(df.columns)}")
print(f"  Data Types:\n{df.dtypes}")

print(f"\n[Missing Values]")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values!")

print(f"\n[Statistical Summary]")
print(df.describe())

print(f"\n[Categorical Variables]")
print(f"  Fuel Types: {df['Fuel_Type'].unique()}")
print(f"  Selling Types: {df['Selling_type'].unique()}")
print(f"  Transmissions: {df['Transmission'].unique()}")
print(f"  Owner Count: {df['Owner'].unique()}")

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n" + "=" * 70)
print("[Step 2] Exploratory Data Analysis...")

# Price statistics
print(f"\n[Selling Price Analysis]")
print(f"  Mean: ₹{df['Selling_Price'].mean():.2f} Lakhs")
print(f"  Median: ₹{df['Selling_Price'].median():.2f} Lakhs")
print(f"  Std Dev: ₹{df['Selling_Price'].std():.2f} Lakhs")
print(f"  Min: ₹{df['Selling_Price'].min():.2f} Lakhs")
print(f"  Max: ₹{df['Selling_Price'].max():.2f} Lakhs")

# Year analysis
print(f"\n[Year Analysis]")
print(f"  Earliest Year: {df['Year'].min()}")
print(f"  Latest Year: {df['Year'].max()}")
print(f"  Cars per Year:\n{df['Year'].value_counts().sort_index().tail()}")

# Fuel type analysis
print(f"\n[Fuel Type Distribution]")
print(df['Fuel_Type'].value_counts())

# Create visualizations
# 1. Price distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Car Price Distribution and Relationships', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Selling_Price'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Selling Price (Lakhs)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Distribution of Car Selling Prices')
axes[0, 0].grid(alpha=0.3)

# 2. Price vs Year
axes[0, 1].scatter(df['Year'], df['Selling_Price'], alpha=0.6, color='#e74c3c', s=50)
axes[0, 1].set_xlabel('Year', fontweight='bold')
axes[0, 1].set_ylabel('Selling Price (Lakhs)', fontweight='bold')
axes[0, 1].set_title('Car Price vs Year of Manufacture')
axes[0, 1].grid(alpha=0.3)

# 3. Price vs Driven KMs
axes[1, 0].scatter(df['Driven_kms'], df['Selling_Price'], alpha=0.6, color='#2ecc71', s=50)
axes[1, 0].set_xlabel('Kilometers Driven', fontweight='bold')
axes[1, 0].set_ylabel('Selling Price (Lakhs)', fontweight='bold')
axes[1, 0].set_title('Car Price vs Kilometers Driven')
axes[1, 0].grid(alpha=0.3)

# 4. Price by Fuel Type
fuel_price = df.groupby('Fuel_Type')['Selling_Price'].mean()
axes[1, 1].bar(fuel_price.index, fuel_price.values, color=['#9b59b6', '#e67e22', '#1abc9c'], 
               edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Average Selling Price (Lakhs)', fontweight='bold')
axes[1, 1].set_title('Average Price by Fuel Type')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_Price_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Price analysis plots saved")
plt.close()

# Correlation analysis
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/02_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Correlation matrix saved")
plt.close()

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

print("\n" + "=" * 70)
print("[Step 3] Data Preprocessing...")

df_processed = df.copy()

# Encode categorical variables
le_fuel = LabelEncoder()
le_selling = LabelEncoder()
le_transmission = LabelEncoder()

df_processed['Fuel_Type'] = le_fuel.fit_transform(df_processed['Fuel_Type'])
df_processed['Selling_type'] = le_selling.fit_transform(df_processed['Selling_type'])
df_processed['Transmission'] = le_transmission.fit_transform(df_processed['Transmission'])

# Drop car name as it's not useful for prediction
df_processed = df_processed.drop('Car_Name', axis=1)

# Prepare features and target
X = df_processed.drop('Selling_Price', axis=1)
y = df_processed['Selling_Price']

print(f"✓ Data preprocessed and encoded")
print(f"  Features: {list(X.columns)}")
print(f"  Target: Selling_Price")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✓ Data split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")
print(f"  Train-Test Split: 80-20")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")

# ==========================================
# 4. MODEL TRAINING
# ==========================================

print("\n" + "=" * 70)
print("[Step 4] Training Regression Models...")

models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
}

results = {}

for model_name, model in models.items():
    print(f"\n[Training] {model_name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results[model_name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_test': y_pred_test
    }
    
    print(f"  ✓ Train R²: {train_r2:.4f}")
    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ Test RMSE: {test_rmse:.4f}")
    print(f"  ✓ Test MAE: {test_mae:.4f}")
    print(f"  ✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# 5. MODEL EVALUATION
# ==========================================

print("\n" + "=" * 70)
print("[Step 5] Detailed Model Evaluation...")

best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred_test']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"   Test RMSE: {results[best_model_name]['test_rmse']:.4f} Lakhs")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.4f} Lakhs")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\n[Feature Importance]")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

print("\n" + "=" * 70)
print("[Step 6] Creating Visualizations...")

# 1. Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['test_r2', 'test_rmse', 'test_mae', 'cv_mean']
metric_labels = ['R² Score', 'RMSE (Lakhs)', 'MAE (Lakhs)', 'CV Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (ax, metric, label, color) in enumerate(zip(axes.flat, metrics, metric_labels, colors)):
    model_names = list(results.keys())
    values = [results[m][metric] for m in model_names]
    
    bars = ax.bar(range(len(model_names)), values, color=color, edgecolor='black', linewidth=1.5)
    ax.set_xticklabels([name.replace(' ', '\n') for name in model_names], fontsize=9)
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(f'{label} Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/03_Model_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison saved")
plt.close()

# 2. Actual vs Predicted for Best Model
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred_best, alpha=0.6, color='#3498db', s=50, edgecolor='black')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price (Lakhs)', fontweight='bold')
ax.set_ylabel('Predicted Price (Lakhs)', fontweight='bold')
ax.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
print("✓ Actual vs Predicted plot saved")
plt.close()

# 3. Residuals Plot
residuals = y_test - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_pred_best, residuals, alpha=0.6, color='#e74c3c', s=50, edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Price (Lakhs)', fontweight='bold')
axes[0].set_ylabel('Residuals', fontweight='bold')
axes[0].set_title('Residuals vs Predicted Values', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].hist(residuals, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Distribution of Residuals', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_Residuals_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Residuals analysis saved")
plt.close()

# 4. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                   color='#9b59b6', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/06_Feature_Importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved")
    plt.close()

# 5. Prediction Error Distribution
fig, ax = plt.subplots(figsize=(10, 6))
errors = np.abs(y_test - y_pred_best)
ax.hist(errors, bins=20, color='#f39c12', edgecolor='black', alpha=0.7)
ax.set_xlabel('Absolute Error (Lakhs)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold', pad=20)
ax.axvline(errors.mean(), color='red', linestyle='--', lw=2, label=f'Mean Error: ₹{errors.mean():.2f}L')
ax.axvline(errors.median(), color='green', linestyle='--', lw=2, label=f'Median Error: ₹{errors.median():.2f}L')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/07_Error_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Error distribution plot saved")
plt.close()

# ==========================================
# 7. SUMMARY AND INSIGHTS
# ==========================================

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\n📊 MODEL RANKINGS (by Test R²):")
sorted_results = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    print(f"   {rank}. {model_name}")
    print(f"      R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.4f}L | MAE: {metrics['test_mae']:.4f}L")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ✓ Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"   ✓ Test RMSE: {results[best_model_name]['test_rmse']:.4f} Lakhs")
print(f"   ✓ Test MAE: {results[best_model_name]['test_mae']:.4f} Lakhs")
print(f"   ✓ Cross-Validation: {results[best_model_name]['cv_mean']:.4f} ± {results[best_model_name]['cv_std']:.4f}")

print(f"\n📈 DATASET INFORMATION:")
print(f"   • Total Cars: {len(df)}")
print(f"   • Training Samples: {len(X_train)}")
print(f"   • Testing Samples: {len(X_test)}")
print(f"   • Features Used: {X.shape[1]}")
print(f"   • Price Range: ₹{y.min():.2f}L - ₹{y.max():.2f}L")
print(f"   • Average Price: ₹{y.mean():.2f}L")

print(f"\n💡 MODEL INTERPRETATION:")
print(f"   • R² Score: {results[best_model_name]['test_r2']:.1%} of price variance explained")
error_percentage = (results[best_model_name]['test_mae'] / y.mean()) * 100
print(f"   • Average Error: ₹{results[best_model_name]['test_mae']:.2f}L (~{error_percentage:.1f}% of avg price)")
print(f"   • Model Reliability: {'Excellent' if results[best_model_name]['test_r2'] > 0.9 else 'Very Good' if results[best_model_name]['test_r2'] > 0.8 else 'Good'}")

print(f"\n✅ ALL VISUALIZATIONS SAVED:")
print(f"   01. Price Analysis (distribution, relationships)")
print(f"   02. Correlation Matrix (feature relationships)")
print(f"   03. Model Comparison (all 4 models)")
print(f"   04. Actual vs Predicted (model fit)")
print(f"   05. Residuals Analysis (error patterns)")
print(f"   06. Feature Importance (key drivers)")
print(f"   07. Error Distribution (prediction accuracy)")

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nAuthor: Pragyan Kumar Beheruk")
print("Repository: OIBSIP")
print("Task: Car Price Prediction - Task 3")
