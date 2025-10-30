# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns  # For confusion matrix heatmap

try:
    # Step 1: Load the dataset
    # Using Iris dataset: Classify flowers into 3 categories (setosa, versicolor, virginica) based on features.
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    print("Dataset loaded. Features:", iris['feature_names'])
    print("Classes:", iris['target_names'])
    print("First 5 rows:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Step 2: Preprocess the dataset
    # Separate features (X) and target (y).
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize features (mean=0, std=1) for KNN, as it's distance-based.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train KNN models with different K values and evaluate
    k_values = [1, 3, 5, 7, 9]  # Different K values to compare
    results = []  # Store results for comparison

    for k in k_values:
        # Train the model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate: Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Evaluate: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Evaluate: Precision, Recall, F1-Score (macro-averaged for multi-class)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Store results
        results.append({
            'K': k,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\nK={k}:")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Precision (macro): {precision:.2f}")
        print(f"  Recall (macro): {recall:.2f}")
        print(f"  F1-Score (macro): {f1:.2f}")
        print("  Confusion Matrix:")
        print(cm)
        
        # Optional: Plot confusion matrix for K=5 as an example
        if k == 5:
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris['target_names'], yticklabels=iris['target_names'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for K={k}')
            plt.show()

    # Step 5: Compare results across K values
    results_df = pd.DataFrame(results)
    print("\nComparison of Results Across K Values:")
    print(results_df)
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        plt.plot(results_df['K'], results_df[metric], marker='o', label=metric)
    plt.xlabel('K Value')
    plt.ylabel('Score')
    plt.title('KNN Performance Metrics vs. K Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nKNN implementation complete! Optimal K often balances bias/variance (e.g., K=5 is common).")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")




#Output:


https://drive.google.com/file/d/12Eo_nPjq193vLEFEcuiKM2kTKp9n0S_f/view?usp=drivesdk


