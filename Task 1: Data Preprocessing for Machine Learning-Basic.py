# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns  # For loading the Titanic dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer  # KNN for advanced imputation
from sklearn.ensemble import RandomForestClassifier  # For a simple model example
from sklearn.metrics import accuracy_score

try:
    # Step 1: Load the raw dataset
    # Using the Titanic dataset (common for ML preprocessing demos). It has missing values, categoricals, and numericals.
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    titanic = sns.load_dataset('titanic')
    # Select relevant columns for simplicity (you can adjust).
    data = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()
    print("Original dataset (first 10 rows):")
    print(data.head(10))
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values per column:\n{data.isnull().sum()}")

    # Step 2: Handle missing data
    # - 'age' has many NaNs: Fill with median (robust to outliers).
    # - 'embarked' has few NaNs: Fill with mode (most frequent).
    # - 'fare' has 1 NaN: Fill with mean.
    # - Drop rows if more than 50% missing (none here, but good practice).
    # Advanced option: Use KNN imputation for 'age' (uncomment if preferred).
    # imputer_age = KNNImputer(n_neighbors=5)
    # data[['age']] = imputer_age.fit_transform(data[['age']])

    # Fill 'age' with median (use assignment instead of inplace for pandas compatibility)
    data['age'] = data['age'].fillna(data['age'].median())
    # Fill 'embarked' with mode
    data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
    # Fill 'fare' with mean
    data['fare'] = data['fare'].fillna(data['fare'].mean())

    # Optional: Drop rows with excessive missing values (threshold: 50% of columns)
    threshold = len(data.columns) * 0.5
    data = data.dropna(thresh=threshold)
    if data.empty:
        raise ValueError("Dataset became empty after dropping rows. Check your threshold or data.")

    print("\nAfter handling missing data:")
    print(data.head(10))
    print(f"Missing values after imputation:\n{data.isnull().sum()}")

    # Step 3: Feature engineering (optional but common in preprocessing)
    # Create 'family_size' as a new numerical feature (sibsp + parch + 1).
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    # Drop original 'sibsp' and 'parch' to avoid redundancy.
    data = data.drop(['sibsp', 'parch'], axis=1)

    # Handle outliers (e.g., cap 'fare' at 99th percentile to reduce skewness).
    fare_cap = data['fare'].quantile(0.99)
    data['fare'] = np.where(data['fare'] > fare_cap, fare_cap, data['fare'])

    print("\nAfter feature engineering and outlier handling:")
    print(data.head(10))

    # Step 4: Encode categorical variables
    # 'sex' and 'embarked' are categorical. Use one-hot encoding for non-ordinal categories.
    # 'pclass' is ordinal (1,2,3), so label encoding could work, but we'll one-hot for consistency.
    categorical_cols = ['sex', 'embarked', 'pclass']
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for sklearn 1.2+
    encoded_features = onehot_encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(categorical_cols))
    # Reset indices to avoid misalignment
    data = data.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    data = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1, ignore_index=False)

    # Alternative: Label encoding for ordinal (uncomment if needed for 'pclass').
    # label_encoder = LabelEncoder()
    # data['pclass'] = label_encoder.fit_transform(data['pclass'])

    print("\nAfter encoding categorical variables (one-hot):")
    print(data.head(10))

    # Step 5: Normalize or standardize numerical features
    # Numerical features: 'age', 'fare', 'family_size'.
    # Standardize (z-score) for algorithms sensitive to scale (e.g., logistic regression).
    numerical_cols = ['age', 'fare', 'family_size']
    scaler = StandardScaler()  # Use MinMaxScaler() for normalization (0-1 scale) if preferred.
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    # Ensure numerical columns are float type after scaling
    data[numerical_cols] = data[numerical_cols].astype(float)

    print("\nAfter standardizing numerical features:")
    print(data.head(10))

    # Step 6: Split the dataset into training and testing sets
    # 'survived' is the target. Split 80/20, stratify to maintain class balance.
    X = data.drop('survived', axis=1)
    y = data['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Validation: Ensure no missing values and shapes are correct
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected after preprocessing. Check imputation.")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Train/test split resulted in empty sets. Check data size.")

    # Step 7: Develop a simple ML model (optional extension to show pipeline usage)
    # Train a Random Forest classifier on the preprocessed data.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy on test set: {accuracy:.2f}")
    print("Preprocessing and basic model development complete! The data is ready for further ML tasks.")

except Exception as e:
    print(f"An error occurred: {e}. Please check your data or library versions.")











#Output:


Original dataset (first 10 rows):
   survived  pclass     sex  ...  parch     fare  embarked
0         0       3    male  ...      0   7.2500         S
1         1       1  female  ...      0  71.2833         C
2         1       3  female  ...      0   7.9250         S
3         1       1  female  ...      0  53.1000         S
4         0       3    male  ...      0   8.0500         S
5         0       3    male  ...      0   8.4583         Q
6         0       1    male  ...      0  51.8625         S
7         0       3    male  ...      1  21.0750         S
8         1       3  female  ...      2  11.1333         S
9         1       2  female  ...      0  30.0708         C

[10 rows x 8 columns]
Shape: (891, 8)
Data types:
survived      int64
pclass        int64
sex          object
age         float64
sibsp         int64
parch         int64
fare        float64
embarked     object
dtype: object
Missing values per column:
survived      0
pclass        0
sex           0
age         177
sibsp         0
parch         0
fare          0
embarked      2
dtype: int64

After handling missing data:
   survived  pclass     sex  ...  parch     fare  embarked
0         0       3    male  ...      0   7.2500         S
1         1       1  female  ...      0  71.2833         C
2         1       3  female  ...      0   7.9250         S
3         1       1  female  ...      0  53.1000         S
4         0       3    male  ...      0   8.0500         S
5         0       3    male  ...      0   8.4583         Q
6         0       1    male  ...      0  51.8625         S
7         0       3    male  ...      1  21.0750         S
8         1       3  female  ...      2  11.1333         S
9         1       2  female  ...      0  30.0708         C

[10 rows x 8 columns]
Missing values after imputation:
survived    0
pclass      0
sex         0
age         0
sibsp       0
parch       0
fare        0
embarked    0
dtype: int64

After feature engineering and outlier handling:
   survived  pclass     sex  ...     fare  embarked family_size0         0       3    male  ...   7.2500         S           21         1       1  female  ...  71.2833         C           22         1       3  female  ...   7.9250         S           13         1       1  female  ...  53.1000         S           24         0       3    male  ...   8.0500         S           15         0       3    male  ...   8.4583         Q           16         0       1    male  ...  51.8625         S           17         0       3    male  ...  21.0750         S           58         1       3  female  ...  11.1333         S           39         1       2  female  ...  30.0708         C           2
[10 rows x 7 columns]

After encoding categorical variables (one-hot):
   survived   age     fare  ...  embarked_S  pclass_2  pclass_30         0  22.0   7.2500  ...         1.0       0.0       1.01         1  38.0  71.2833  ...         0.0       0.0       0.02         1  26.0   7.9250  ...         1.0       0.0       1.03         1  35.0  53.1000  ...         1.0       0.0       0.04         0  35.0   8.0500  ...         1.0       0.0       1.05         0  28.0   8.4583  ...         0.0       0.0       1.06         0  54.0  51.8625  ...         1.0       0.0       0.07         0   2.0  21.0750  ...         1.0       0.0       1.08         1  27.0  11.1333  ...         1.0       0.0       1.09         1  14.0  30.0708  ...         0.0       1.0       0.0
[10 rows x 9 columns]

After standardizing numerical features:
   survived       age  ...  pclass_2  pclass_3
0         0 -0.565736  ...       0.0       1.0
1         1  0.663861  ...       0.0       0.0
2         1 -0.258337  ...       0.0       1.0
3         1  0.433312  ...       0.0       0.0
4         0  0.433312  ...       0.0       1.0
5         0 -0.104637  ...       0.0       1.0
6         0  1.893459  ...       0.0       0.0
7         0 -2.102733  ...       0.0       1.0
8         1 -0.181487  ...       0.0       1.0
9         1 -1.180535  ...       1.0       0.0

[10 rows x 9 columns]

Training set shape: X_train=(712, 8), y_train=(712,)
Testing set shape: X_test=(179, 8), y_test=(179,)

Model accuracy on test set: 0.79
Preprocessing and basic model development complete! The data is ready for further ML tasks.

[Program finished]

                                                                     
