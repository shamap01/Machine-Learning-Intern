# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  # For confusion matrix heatmap

try:
    # Step 1: Load the dataset
    # Using Breast Cancer Wisconsin dataset: Predict binary outcome (malignant=1, benign=0) based on tumor features.
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    cancer = load_breast_cancer()
    data = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']], columns=list(cancer['feature_names']) + ['target'])
    print("Dataset loaded. Features:", list(cancer['feature_names']))
    print("Classes:", cancer['target_names'])  # 0: benign, 1: malignant
    print("First 5 rows:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Step 2: Preprocess the dataset
    # Separate features (X) and target (y: binary outcome).
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize features (mean=0, std=1) for logistic regression stability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Split into training and testing sets (80/20 split, stratified for balance)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter for convergence
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")

    # Step 5: Interpret model coefficients and odds ratios
    # Coefficients show the log-odds change per unit increase in feature.
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],  # Flatten for binary classification
        'Odds Ratio': np.exp(model.coef_[0])  # exp(coef) = odds ratio
    })
    print("\nModel Coefficients and Odds Ratios:")
    print(coefficients.sort_values(by='Coefficient', ascending=False))
    print(f"Intercept (log-odds baseline): {model.intercept_[0]:.2f}")
    # Interpretation:
    # - Coefficient: Positive means feature increases likelihood of malignant (target=1).
    # - Odds Ratio: >1 means higher odds of malignant with feature increase; <1 means lower odds.
    # Example: If 'worst radius' has coef=1.5, odds ratio=4.48 → 4.48x higher odds of malignant per unit increase.

    # Step 6: Evaluate the model
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC

    # Metrics: Accuracy, Precision, Recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer['target_names']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer['target_names'], yticklabels=cancer['target_names'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc:.2f}")
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print("\nLogistic regression implementation complete! For improvements, consider regularization or feature selection.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")


