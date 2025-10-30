#1st Method:-


#Step 1: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


#Step 2: Load and Prepare Dataset


# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#Step 3: Train Decision Tree

# Train with pruning (max_depth to prevent overfitting)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)


#Step 4: Visualize Tree Structure

plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()


#Step 5: Evaluate the Model

Predict and evaluate
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("✅ Model Evaluation:")
print("Accuracy:", accuracy)
print("F1 Score (macro):", f1)









#2nd Method:-


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

try:
    # Step 1: Load the dataset
    # Using Iris dataset: Predict flower species (setosa, versicolor, virginica) based on features.
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

    # Step 2: Preprocess the dataset (minimal for Iris)
    # Separate features (X) and target (y).
    X = data.drop('target', axis=1)
    y = data['target']

    # No standardization needed for decision trees, as they are invariant to scale.

    # Step 3: Split into training and testing sets (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train an initial decision tree (unpruned, to show overfitting potential)
    model_unpruned = DecisionTreeClassifier(random_state=42)  # Default: no pruning
    model_unpruned.fit(X_train, y_train)
    print("\nUnpruned model trained. Depth:", model_unpruned.get_depth())

    # Step 5: Visualize the tree structure (for the unpruned model)
    plt.figure(figsize=(20, 10))
    plot_tree(model_unpruned, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True, rounded=True)
    plt.title("Unpruned Decision Tree")
    plt.show()

    # Step 6: Prune the tree to prevent overfitting
    # Method 1: Pre-pruning with max_depth (simpler, limits depth to 3)
    model_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_pruned.fit(X_train, y_train)
    print("\nPruned model (max_depth=3) trained. Depth:", model_pruned.get_depth())

    # Alternative: Post-pruning with cost complexity (ccp_alpha) - uncomment to use
    # path = model_unpruned.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas = path.ccp_alphas
    # model_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alphas[-2])  # Choose a moderate alpha
    # model_pruned.fit(X_train, y_train)
    # print("Pruned model (ccp_alpha) trained.")

    # Visualize the pruned tree
    plt.figure(figsize=(12, 8))
    plot_tree(model_pruned, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True, rounded=True)
    plt.title("Pruned Decision Tree (max_depth=3)")
    plt.show()

    # Step 7: Evaluate the pruned model
    # Predict on test set
    y_pred = model_pruned.predict(X_test)

    # Metrics: Accuracy and F1-Score (macro-averaged for multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"F1-Score (macro): {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris['target_names']))

    print("\nDecision tree implementation complete! Pruning helps generalize to unseen data.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")





