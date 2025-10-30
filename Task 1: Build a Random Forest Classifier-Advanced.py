# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
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

    # Step 2: Preprocess the dataset (minimal for this dataset)
    # Separate features (X) and target (y).
    X = data.drop('target', axis=1)
    y = data['target']

    # No scaling needed for Random Forest, as it's tree-based and invariant to scale.

    # Step 3: Split into training and testing sets (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train a Random Forest model and tune hyperparameters
    # Initial model with default parameters
    rf_initial = RandomForestClassifier(random_state=42)
    rf_initial.fit(X_train, y_train)
    print("\nInitial Random Forest trained (default params).")

    # Hyperparameter tuning using GridSearchCV (tune n_estimators and max_depth)
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [None, 10, 20, 30]  # Maximum depth of trees
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.2f}")

    # Step 5: Evaluate the tuned model using cross-validation and classification metrics
    # Cross-validation on training set (5-fold)
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1')
    print(f"\nCross-validation F1-scores: {cv_scores}")
    print(f"Mean CV F1-score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nTest Set Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer['target_names']))

    # Step 6: Perform feature importance analysis
    # Extract feature importances from the best model
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importances.head(10))

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances in Random Forest')
    plt.gca().invert_yaxis()  # Highest at top
    plt.show()

    print("\nRandom Forest implementation complete! Feature importances help identify key predictors (e.g., for model simplification).")

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

Training set shape: X_train=(455, 30), y_train=(455,)
Testing set shape: X_test=(114, 30), y_test=(114,)


Initial Random Forest trained (default params).

Best hyperparameters: {'max_depth': None, 'n_estimators': 200}
Best cross-validation F1-score: 0.97

Cross-validation F1-scores: [0.97391304 0.99130435 0.94827586 0.95575221 0.97297297]
Mean CV F1-score: 0.97 (+/- 0.03)

Test Set Metrics:
Precision: 0.96
Recall: 0.97
F1-Score: 0.97

Classification Report:
              precision    recall  f1-score   support

   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114


Top 10 Feature Importances:
                 Feature  Importance
22       worst perimeter    0.133100
23            worst area    0.128052
27  worst concave points    0.108107
7    mean concave points    0.094414
20          worst radius    0.090639
0            mean radius    0.058662
2         mean perimeter    0.055242
3              mean area    0.049938
6         mean concavity    0.046207
26       worst concavity    0.035357

Random Forest implementation complete! Feature importances help identify key predictors (e.g., for model simplification).



https://drive.google.com/file/d/1jJiac-FWa1KmXTho-0LuFTxNN_3jGhPp/view?usp=drivesdk


