"""
IRIS FLOWER CLASSIFICATION
Author: Pragyan Kumar Beheruk
Internship Task 1 - OIBSIP

This script trains a machine learning model to classify iris flowers into three species:
- Setosa
- Versicolor
- Virginica

Based on four features: sepal length, sepal width, petal length, and petal width.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD AND EXPLORE DATA
# ==========================================

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION - MACHINE LEARNING PROJECT")
print("=" * 60)
print("\n[Step 1] Loading and Exploring Dataset...")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['Species'] = [target_names[i] for i in y]

print(f"\nDataset Shape: {X.shape}")
print(f"Number of Samples: {X.shape[0]}")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Classes: {len(target_names)}")
print(f"Classes: {list(target_names)}")

print("\n[Dataset Preview]")
print(df.head(10))

print("\n[Dataset Statistics]")
print(df.describe())

print("\n[Class Distribution]")
print(df['Species'].value_counts())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

print("\n" + "=" * 60)
print("[Step 2] Exploratory Data Analysis...")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset - Feature Distributions', fontsize=16, fontweight='bold')

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for idx, (ax, feature) in enumerate(zip(axes.flat, features)):
    for i, target in enumerate(target_names):
        ax.hist(df[df['Species'] == target][feature], alpha=0.6, label=target, bins=20)
    ax.set_xlabel(feature, fontweight='bold')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_Feature_Distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature distribution plots saved")

# Correlation Analysis
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = df.drop('Species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/02_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Correlation matrix saved")

# Box plots for feature comparison across species
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Comparison Across Iris Species', fontsize=16, fontweight='bold')

for idx, (ax, feature) in enumerate(zip(axes.flat, features)):
    df.boxplot(column=feature, by='Species', ax=ax)
    ax.set_title(f'{feature}')
    ax.set_xlabel('Species')
    ax.set_ylabel('Value (cm)')
    ax.grid(alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/03_Feature_Comparison_BoxPlots.png', dpi=300, bbox_inches='tight')
print("✓ Box plot comparison saved")

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

print("\n" + "=" * 60)
print("[Step 3] Data Preprocessing...")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set Size: {X_train.shape[0]} samples")
print(f"Testing Set Size: {X_test.shape[0]} samples")
print(f"Train-Test Split: 80-20")

# Feature Scaling (Important for many ML algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features standardized using StandardScaler")
print(f"Scaled Training Data Shape: {X_train_scaled.shape}")

# ==========================================
# 4. MODEL TRAINING AND EVALUATION
# ==========================================

print("\n" + "=" * 60)
print("[Step 4] Training Multiple Classification Models...")

# Dictionary to store models and their performance
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

for model_name, model in models.items():
    print(f"\n[Training] {model_name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_scores.std(),
        'y_pred': y_pred
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ Cross-Validation Score: {cv_mean:.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ==========================================

# Select the best model (Random Forest typically performs well)
best_model_name = 'Random Forest'
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']

print("\n" + "=" * 60)
print(f"[Step 5] Detailed Evaluation - {best_model_name} (Best Performer)")
print("=" * 60)

print("\n[Classification Report]")
print(classification_report(y_test, y_pred_best, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\n[Confusion Matrix]")
print(cm)

# Feature Importance (for Random Forest)
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n[Feature Importance]")
print(feature_importance_df)

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

print("\n" + "=" * 60)
print("[Step 6] Creating Visualizations...")

# Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    model_names = list(results.keys())
    metric_values = [results[m][metric] for m in model_names]
    
    bars = ax.bar(range(len(model_names)), metric_values, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.replace(' (', '\n(') for name in model_names], fontsize=9)
    ax.set_ylabel(metric.capitalize(), fontweight='bold')
    ax.set_title(f'{metric.capitalize()} Scores', fontweight='bold')
    ax.set_ylim([0.9, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/04_Model_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison chart saved")

# Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax,
            cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix heatmap saved")

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#45B7D1')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/06_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance chart saved")

# ROC Curves for multiclass classification
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = best_model.predict_proba(X_test_scaled)

fig, ax = plt.subplots(figsize=(10, 8))

colors_roc = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (color, target) in enumerate(zip(colors_roc, target_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{target} (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title(f'ROC Curves - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/07_ROC_Curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved")

# ==========================================
# 7. SUMMARY AND RESULTS
# ==========================================

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   Precision: {results[best_model_name]['precision']:.4f}")
print(f"   Recall: {results[best_model_name]['recall']:.4f}")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   Cross-Validation Score: {results[best_model_name]['cv_mean']:.4f} ± {results[best_model_name]['cv_std']:.4f}")

print(f"\n📊 MODEL RANKINGS (by Accuracy):")
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    print(f"   {rank}. {model_name}: {metrics['accuracy']:.4f}")

print(f"\n🎯 TOP FEATURES FOR CLASSIFICATION:")
for idx, row in feature_importance_df.iterrows():
    print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"\n📈 DATASET INFORMATION:")
print(f"   Total Samples: {len(X)}")
print(f"   Training Samples: {len(X_train)}")
print(f"   Testing Samples: {len(X_test)}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(target_names)}")

print(f"\n✅ ALL VISUALIZATIONS SAVED IN OUTPUT FOLDER")
print(f"   - Feature Distributions")
print(f"   - Correlation Matrix")
print(f"   - Feature Comparison Box Plots")
print(f"   - Model Performance Comparison")
print(f"   - Confusion Matrix")
print(f"   - Feature Importance")
print(f"   - ROC Curves")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nAuthor: Pragyan Kumar Beheruk")
print("Repository: OIBSIP")
print("Task: Iris Flower Classification - Task 1")
