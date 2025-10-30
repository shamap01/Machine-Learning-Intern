# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Sample dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the raw dataset
# Using the Iris dataset as an example (it has 4 numerical features and 1 categorical target).
# In a real scenario, replace this with pd.read_csv('your_file.csv') for your own data.
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['species'])

# Introduce some missing values artificially for demonstration (e.g., set some to NaN)
data.iloc[0:5, 0] = np.nan  # Missing values in the first column
data.iloc[10:15, 2] = np.nan  # Missing values in the third column
print("Original dataset (first 10 rows):")
print(data.head(10))
print(f"Shape: {data.shape}")

# Step 2: Handle missing data
# Option 1: Fill with mean (for numerical columns) or median (if outliers are present).
# We'll use mean for simplicity. You can switch to median by changing 'mean' to 'median'.
imputer = SimpleImputer(strategy='mean')  # Fills NaN with column mean
numerical_cols = data.select_dtypes(include=[np.number]).columns  # Identify numerical columns
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Option 2: Alternatively, drop rows with missing values (uncomment if preferred).
# data = data.dropna()

print("\nAfter handling missing data (filled with mean):")
print(data.head(10))

# Step 3: Encode categorical variables
# The 'species' column is categorical (0, 1, 2 representing classes). We'll use label encoding for simplicity.
# If you have string categories, use one-hot encoding instead.
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Alternative: One-hot encoding (uncomment if needed for multi-class or string categories).
# onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
# encoded_species = onehot_encoder.fit_transform(data[['species']])
# encoded_df = pd.DataFrame(encoded_species, columns=onehot_encoder.get_feature_names_out(['species']))
# data = pd.concat([data.drop('species', axis=1), encoded_df], axis=1)

print("\nAfter encoding categorical variables (label encoding on 'species'):")
print(data.head(10))

# Step 4: Normalize or standardize numerical features
# Numerical features: sepal length, sepal width, petal length, petal width.
# Standardization (z-score) is common for algorithms like SVM/PCA; normalization (min-max) for neural networks.
scaler = StandardScaler()  # Use MinMaxScaler() for normalization instead
data[numerical_cols[:-1]] = scaler.fit_transform(data[numerical_cols[:-1]])  # Exclude 'species' if it's the target

print("\nAfter standardizing numerical features:")
print(data.head(10))

# Step 5: Split the dataset into training and testing sets
# Assume 'species' is the target variable. Adjust column names as needed for your dataset.
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target

# Split: 80% training, 20% testing. Use random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
print("\nPreprocessing complete! The data is now ready for machine learning models.")








#output:


Original dataset (first 10 rows):
   sepal length (cm)  ...  species
0                NaN  ...      0.0
1                NaN  ...      0.0
2                NaN  ...      0.0
3                NaN  ...      0.0
4                NaN  ...      0.0
5                5.4  ...      0.0
6                4.6  ...      0.0
7                5.0  ...      0.0
8                4.4  ...      0.0
9                4.9  ...      0.0

[10 rows x 5 columns]
Shape: (150, 5)

After handling missing data (filled with mean):
   sepal length (cm)  ...  species
0           5.877241  ...      0.0
1           5.877241  ...      0.0
2           5.877241  ...      0.0
3           5.877241  ...      0.0
4           5.877241  ...      0.0
5           5.400000  ...      0.0
6           4.600000  ...      0.0
7           5.000000  ...      0.0
8           4.400000  ...      0.0
9           4.900000  ...      0.0

[10 rows x 5 columns]

After encoding categorical variables (label encoding on 'species'):
   sepal length (cm)  ...  species
0           5.877241  ...        0
1           5.877241  ...        0
2           5.877241  ...        0
3           5.877241  ...        0
4           5.877241  ...        0
5           5.400000  ...        0
6           4.600000  ...        0
7           5.000000  ...        0
8           4.400000  ...        0
9           4.900000  ...        0

[10 rows x 5 columns]

After standardizing numerical features:
   sepal length (cm)  ...  species
0       1.104514e-15  ...        0
1       1.104514e-15  ...        0
2       1.104514e-15  ...        0
3       1.104514e-15  ...        0
4       1.104514e-15  ...        0
5      -5.934844e-01  ...        0
6      -1.588343e+00  ...        0
7      -1.090913e+00  ...        0
8      -1.837057e+00  ...        0
9      -1.215271e+00  ...        0

[10 rows x 5 columns]

Training set shape: X_train=(120, 4), y_train=(120,)
Testing set shape: X_test=(30, 4), y_test=(30,)

Preprocessing complete! The data is now ready for machine learning models.

[Program finished]
