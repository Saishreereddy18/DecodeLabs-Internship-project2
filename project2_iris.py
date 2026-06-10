# ============================================
# DecodeLabs - Project 2: Data Classification
# Iris Dataset using KNN Algorithm
# ============================================

# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ============================================
# STEP 1: LOAD THE DATASET
# ============================================
iris = load_iris()
X = iris.data
y = iris.target

print("=" * 50)
print("STEP 1: Dataset Loaded")
print("=" * 50)
print(f"Total samples : {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Classes       : {iris.target_names}")
print()

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)
print(df.head(10))
print()

# ============================================
# STEP 2: SCALE THE FEATURES
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 50)
print("STEP 2: Features Scaled (mean=0, variance=1)")
print("=" * 50)
print(f"Before scaling - Mean: {X[:,0].mean():.2f}, Std: {X[:,0].std():.2f}")
print(f"After  scaling - Mean: {X_scaled[:,0].mean():.2f}, Std: {X_scaled[:,0].std():.2f}")
print()

# ============================================
# STEP 3: SPLIT INTO TRAIN AND TEST
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("=" * 50)
print("STEP 3: Data Split")
print("=" * 50)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print()

# ============================================
# STEP 4: TRAIN THE KNN MODEL
# ============================================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("=" * 50)
print("STEP 4: Model Trained (KNN, k=5)")
print("=" * 50)
print("Model is ready!")
print()

# ============================================
# STEP 5: MAKE PREDICTIONS
# ============================================
predictions = model.predict(X_test)

print("=" * 50)
print("STEP 5: Predictions Made")
print("=" * 50)
print(f"Predicted : {predictions}")
print(f"Actual    : {y_test}")
print()

# ============================================
# STEP 6: EVALUATE THE MODEL
# ============================================
cm = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print("=" * 50)
print("STEP 6: Model Evaluation")
print("=" * 50)
print("Confusion Matrix:")
print(cm)
print()
print(f"F1 Score (weighted): {f1:.4f}")
print()
print("Detailed Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ============================================
# BONUS: PLOT CONFUSION MATRIX
# ============================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris KNN Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix plot saved as confusion_matrix.png")