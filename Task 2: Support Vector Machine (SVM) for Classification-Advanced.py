#1st Method:-


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA  # For 2D visualization
import matplotlib.pyplot as plt

try:
    # Step 1: Load the dataset
    # Using Breast Cancer Wisconsin dataset: Predict binary outcome (malignant=1, benign=0) based on 30 features.
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
    # Separate features (X) and target (y).
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize features (mean=0, std=1) for SVM, as it's sensitive to scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Split into training and testing sets (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train SVM models with different kernels and compare performance
    kernels = ['linear', 'rbf']  # Linear for linear separability, RBF for non-linear
    results = []

    for kernel in kernels:
        # Train the SVM model
        svm_model = SVC(kernel=kernel, random_state=42, probability=True)  # probability=True for AUC
        svm_model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = svm_model.predict(X_test)
        y_pred_proba = svm_model.predict_proba(X_test)[:, 1]  # For AUC
        
        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Kernel': kernel,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc
        })
        
        print(f"\nSVM with {kernel} kernel:")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  AUC: {auc:.2f}")

    # Compare results
    results_df = pd.DataFrame(results)
    print("\nComparison of SVM Kernels:")
    print(results_df)

    # Step 5: Visualize the decision boundary (using 2D PCA projection for the best kernel)
    # Choose the best kernel based on AUC (e.g., RBF often performs better)
    best_kernel = results_df.loc[results_df['AUC'].idxmax(), 'Kernel']
    print(f"\nVisualizing decision boundary for best kernel: {best_kernel}")

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train SVM on PCA-reduced data for visualization
    svm_vis = SVC(kernel=best_kernel, random_state=42)
    svm_vis.fit(X_train_pca, y_train)

    # Create a mesh grid for decision boundary
    h = 0.02  # Step size
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', label='Train')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', marker='x', s=100, label='Test')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'SVM Decision Boundary ({best_kernel} kernel)')
    plt.legend()
    plt.show()

    print("\nSVM implementation complete! RBF kernel often handles non-linear data better than linear.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")









#Output:


Dataset loaded. Features: [np.str_('mean radius'), np.str_('mean texture'), np.str_('mean perimeter'), np.str_('mean area'), np.str_('mean smoothness'), np.str_('mean compactness'), np.str_('mean concavity'), np.str_('mean concave points'), np.str_('mean symmetry'), np.str_('mean fractal dimension'), np.str_('radius error'), np.str_('texture error'), np.str_('perimeter error'), np.str_('area error'), np.str_('smoothness error'), np.str_('compactness error'), np.str_('concavity error'), np.str_('concave points error'), np.str_('symmetry error'), np.str_('fractal dimension error'), np.str_('worst radius'), np.str_('worst texture'), np.str_('worst perimeter'), np.str_('worst area'), np.str_('worst smoothness'), np.str_('worst compactness'), np.str_('worst concavity'), np.str_('worst concave points'), np.str_('worst symmetry'), np.str_('worst fractal dimension')]
Classes: ['malignant' 'benign']
First 5 rows:
   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \
0        17.99         10.38          122.80     1001.0          0.11840   
1        20.57         17.77          132.90     1326.0          0.08474   
2        19.69         21.25          130.00     1203.0          0.10960   
3        11.42         20.38           77.58      386.1          0.14250   
4        20.29         14.34          135.10     1297.0          0.10030   

   mean compactness  mean concavity  mean concave points  mean symmetry  \
0           0.27760          0.3001              0.14710         0.2419   
1           0.07864          0.0869              0.07017         0.1812   
2           0.15990          0.1974              0.12790         0.2069   
3           0.28390          0.2414              0.10520         0.2597   
4           0.13280          0.1980              0.10430         0.1809   

   mean fractal dimension  radius error  texture error  perimeter error  \
0                 0.07871        1.0950         0.9053            8.589   
1                 0.05667        0.5435         0.7339            3.398   
2                 0.05999        0.7456         0.7869            4.585   
3                 0.09744        0.4956         1.1560            3.445   
4                 0.05883        0.7572         0.7813            5.438   

   area error  smoothness error  compactness error  concavity error  \
0      153.40          0.006399            0.04904          0.05373   
1       74.08          0.005225            0.01308          0.01860   
2       94.03          0.006150            0.04006          0.03832   
3       27.23          0.009110            0.07458          0.05661   
4       94.44          0.011490            0.02461          0.05688   

   concave points error  symmetry error  fractal dimension error  \
0               0.01587         0.03003                 0.006193   
1               0.01340         0.01389                 0.003532   
2               0.02058         0.02250                 0.004571   
3               0.01867         0.05963                 0.009208   
4               0.01885         0.01756                 0.005115   

   worst radius  worst texture  worst perimeter  worst area  worst smoothness  \
0         25.38          17.33           184.60      2019.0            0.1622   
1         24.99          23.41           158.80      1956.0            0.1238   
2         23.57          25.53           152.50      1709.0            0.1444   
3         14.91          26.50            98.87       567.7            0.2098   
4         22.54          16.67           152.20      1575.0            0.1374   

   worst compactness  worst concavity  worst concave points  worst symmetry  \
0             0.6656           0.7119                0.2654          0.4601   
1             0.1866           0.2416                0.1860          0.2750   
2             0.4245           0.4504                0.2430          0.3613   
3             0.8663           0.6869                0.2575          0.6638   
4             0.2050           0.4000                0.1625          0.2364   

   worst fractal dimension  target  
0                  0.11890     0.0  
1                  0.08902     0.0  
2                  0.08758     0.0  
3                  0.17300     0.0  
4                  0.07678     0.0  
Shape: (569, 31)
Data types:
mean radius                float64
mean texture               float64
mean perimeter             float64
mean area                  float64
mean smoothness            float64
mean compactness           float64
mean concavity             float64
mean concave points        float64
mean symmetry              float64
mean fractal dimension     float64
radius error               float64
texture error              float64
perimeter error            float64
area error                 float64
smoothness error           float64
compactness error          float64
concavity error            float64
concave points error       float64
symmetry error             float64
fractal dimension error    float64
worst radius               float64
worst texture              float64
worst perimeter            float64
worst area                 float64
worst smoothness           float64
worst compactness          float64
worst concavity            float64
worst concave points       float64
worst symmetry             float64
worst fractal dimension    float64
target                     float64
dtype: object
Missing values:
mean radius                0
mean texture               0
mean perimeter             0
mean area                  0
mean smoothness            0
mean compactness           0
mean concavity             0
mean concave points        0
mean symmetry              0
mean fractal dimension     0
radius error               0
texture error              0
perimeter error            0
area error                 0
smoothness error           0
compactness error          0
concavity error            0
concave points error       0
symmetry error             0
fractal dimension error    0
worst radius               0
worst texture              0
worst perimeter            0
worst area                 0
worst smoothness           0
worst compactness          0
worst concavity            0
worst concave points       0
worst symmetry             0
worst fractal dimension    0
target                     0
dtype: int64

After preprocessing (standardization):
   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \
0     1.097064     -2.073335        1.269934   0.984375         1.568466   
1     1.829821     -0.353632        1.685955   1.908708        -0.826962   
2     1.579888      0.456187        1.566503   1.558884         0.942210   
3    -0.768909      0.253732       -0.592687  -0.764464         3.283553   
4     1.750297     -1.151816        1.776573   1.826229         0.280372   

   mean compactness  mean concavity  mean concave points  mean symmetry  \
0          3.283515        2.652874             2.532475       2.217515   
1         -0.487072       -0.023846             0.548144       0.001392   
2          1.052926        1.363478             2.037231       0.939685   
3          3.402909        1.915897             1.451707       2.867383   
4          0.539340        1.371011             1.428493      -0.009560   

   mean fractal dimension  radius error  texture error  perimeter error  \
0                2.255747      2.489734      -0.565265         2.833031   
1               -0.868652      0.499255      -0.876244         0.263327   
2               -0.398008      1.228676      -0.780083         0.850928   
3                4.910919      0.326373      -0.110409         0.286593   
4               -0.562450      1.270543      -0.790244         1.273189   

   area error  smoothness error  compactness error  concavity error  \
0    2.487578         -0.214002           1.316862         0.724026   
1    0.742402         -0.605351          -0.692926        -0.440780   
2    1.181336         -0.297005           0.814974         0.213076   
3   -0.288378          0.689702           2.744280         0.819518   
4    1.190357          1.483067          -0.048520         0.828471   

   concave points error  symmetry error  fractal dimension error  \
0              0.660820        1.148757                 0.907083   
1              0.260162       -0.805450                -0.099444   
2              1.424827        0.237036                 0.293559   
3              1.115007        4.732680                 2.047511   
4              1.144205       -0.361092                 0.499328   

   worst radius  worst texture  worst perimeter  worst area  worst smoothness  \
0      1.886690      -1.359293         2.303601    2.001237          1.307686   
1      1.805927      -0.369203         1.535126    1.890489         -0.375612   
2      1.511870      -0.023974         1.347475    1.456285          0.527407   
3     -0.281464       0.133984        -0.249939   -0.550021          3.394275   
4      1.298575      -1.466770         1.338539    1.220724          0.220556   

   worst compactness  worst concavity  worst concave points  worst symmetry  \
0           2.616665         2.109526              2.296076        2.750622   
1          -0.430444        -0.146749              1.087084       -0.243890   
2           1.082932         0.854974              1.955000        1.152255   
3           3.893397         1.989588              2.175786        6.046041   
4          -0.313395         0.613179              0.729259       -0.868353   

   worst fractal dimension  
0                 1.937015  
1                 0.281190  
2                 0.201391  
3                 4.935010  
4                -0.397100  

Training set shape: X_train=(455, 30), y_train=(455,)
Testing set shape: X_test=(114, 30), y_test=(114,)


SVM with linear kernel:
  Accuracy: 0.97
  Precision: 0.99
  Recall: 0.97
  AUC: 1.00

SVM with rbf kernel:
  Accuracy: 0.98
  Precision: 0.99
  Recall: 0.99
  AUC: 1.00

Comparison of SVM Kernels:
   Kernel  Accuracy  Precision    Recall       AUC
0  linear  0.973684   0.985915  0.972222  0.996362
1     rbf  0.982456   0.986111  0.986111  0.995040

Visualizing decision boundary for best kernel: linear

SVM implementation complete! RBF kernel often handles non-linear data better than linear.



https://drive.google.com/file/d/1amyzerdwSUpcP_y2SJTe1gJXoF7Z-_Vs/view?usp=drivesdk














#2nd Method:-

#Step 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_scor


#Step 2: Simulate a Binary Dataset

# Generate 2D binary classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)


#Step 3: Split and Scale Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Step 4: Train SVM Models

svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)


#Step 5: Evaluate Models

def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'Accuracy': round(accuracy_score(y, y_pred), 2),
        'Precision': round(precision_score(y, y_pred), 2),
        'Recall': round(recall_score(y, y_pred), 2),
        'AUC': round(roc_auc_score(y, y_prob), 2)
    }

results_linear = evaluate(svm_linear, X_test_scaled, y_test)
results_rbf = evaluate(svm_rbf, X_test_scaled, y_test)

print("📊 Linear Kernel:", results_linear)
print("📊 RBF Kernel:", results_rbf)


#Step 6: Visualize Decision Boundaries

def plot_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

plot_boundary(svm_linear, X_test_scaled, y_test, "SVM with Linear Kernel")
plot_boundary(svm_rbf, X_test_scaled, y_test, "SVM with RBF Kernel")






#Output:


Kernel	Accuracy	Precision	Recall	AUC
Linear  	0.99	          0.98	      1.00        	1.00
RBFv    	0.99	          0.98      	1.00	        1.00
Both models perform exceptionally well on this dataset.

Decision boundaries clearly separate the two classes.

AUC of 1.00 indicates perfect classification.

Would you like to apply this to a real-world dataset or explore hyperparameter tuning like C and gamma? I can help you extend it further.


