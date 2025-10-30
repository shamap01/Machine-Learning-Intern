#1st Method:-


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml  # For Boston Housing dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

try:
    # Step 1: Load the dataset
    # Using Boston Housing dataset: Predict median house value (MEDV) based on features like crime rate, rooms, etc.
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    boston = fetch_openml(name='boston', version=1, as_frame=True)
    data = boston.frame
    print("Dataset loaded. Features:", list(data.columns))
    print("First 5 rows:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Step 2: Preprocess the dataset
    # Boston Housing has no missing values or categoricals, but we'll standardize numerical features for better model performance.
    # Separate features (X) and target (y: MEDV).
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    # Standardize features (mean=0, std=1) to improve linear regression stability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame for readability

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")

    # Step 5: Interpret the model coefficients
    # Coefficients show the impact of each feature on the target (MEDV).
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print("\nModel Coefficients (impact on house price):")
    print(coefficients.sort_values(by='Coefficient', ascending=False))
    print(f"Intercept (baseline price): {model.intercept_:.2f}")
    # Interpretation example: A positive coefficient (e.g., RM: number of rooms) means higher values increase price.
    # Negative (e.g., LSTAT: lower status population) means higher values decrease price.

    # Step 6: Evaluate the model
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate R-squared (variance explained, 0-1; higher is better)
    r2 = r2_score(y_test, y_pred)
    print(f"\nR-squared: {r2:.2f} (closer to 1 is better)")

    # Calculate Mean Squared Error (MSE: average squared prediction error; lower is better)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Optional: Visualize predictions vs. actuals
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs. Predicted House Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.show()

    print("\nModel development complete! For improvements, consider regularization (e.g., Ridge) or feature selection.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")







#Output:



https://drive.google.com/file/d/1cI8-Ov-iiWhfcFJ6drXQgYw21TWYWIxU/view?usp=drivesdk


















#2nd Method:-



# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml  # For Boston Housing dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

try:
    # Step 1: Load the dataset
    # Using Boston Housing dataset: Predict median house value (MEDV) based on features like crime rate, rooms, etc.
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    boston = fetch_openml(name='boston', version=1, as_frame=True)
    data = boston.frame
    print("Dataset loaded. Features:", list(data.columns))
    print("First 5 rows:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Step 2: Preprocess the dataset
    # Boston Housing has no missing values or categoricals, but we'll standardize numerical features for better model performance.
    # Separate features (X) and target (y: MEDV).
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    # Standardize features (mean=0, std=1) to improve linear regression stability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame for readability

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Step 4: Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")

    # Step 5: Interpret the model coefficients
    # Coefficients show the impact of each feature on the target (MEDV).
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print("\nModel Coefficients (impact on house price):")
    print(coefficients.sort_values(by='Coefficient', ascending=False))
    print(f"Intercept (baseline price): {model.intercept_:.2f}")
    # Interpretation example: A positive coefficient (e.g., RM: number of rooms) means higher values increase price.
    # Negative (e.g., LSTAT: lower status population) means higher values decrease price.

    # Step 6: Evaluate the model
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate R-squared (variance explained, 0-1; higher is better)
    r2 = r2_score(y_test, y_pred)
    print(f"\nR-squared: {r2:.2f} (closer to 1 is better)")

    # Calculate Mean Squared Error (MSE: average squared prediction error; lower is better)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Optional: Visualize predictions vs. actuals
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs. Predicted House Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.show()

    print("\nModel development complete! For improvements, consider regularization (e.g., Ridge) or feature selection.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")







#Output:


📄 Raw Dataset:
    size_sqft  price_usd
0        750     150000
1        800     160000
2        850     165000
3        900     170000
4        950     175000
5       1000     180000
6       1050     185000
7       1100     190000

📈 Model Coefficients:
Intercept: 68900.00000000003
Slope (size_sqft): 110.99999999999997

📊 Model Evaluation:
R-squared: 0.9735
Mean Squared Error: 2650000.0

[Program finished]

    
    

