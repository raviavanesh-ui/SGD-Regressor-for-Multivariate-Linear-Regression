# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start

STEP 2. Data preparation

STEP 3. Hypothesis Definition

STEP 4. Cost Function

STEP 5. Parameter Update Rule

STEP 6. Iterative Training

STEP 7. Model evaluation

STEP 8. End


## Program:
```
SGD-Regressor-for-Multivariate-Linear-Regression
Name : M AHAMED SAHUL HAMEED 
Register Number : 212224040016
```
```python
# -------------------------------
# Import Required Libraries
# -------------------------------
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load Dataset
# -------------------------------
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["HousingPrice"] = data.target

print(df.head())

# -------------------------------
# Features and Targets
# -------------------------------
X = df.drop(columns=["AveOccup", "HousingPrice"])   # Input features
Y = df[["AveOccup", "HousingPrice"]]                # Output variables

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# -------------------------------
# SGD Regressor Model
# -------------------------------
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

model = MultiOutputRegressor(sgd)
model.fit(X_train, Y_train)

# -------------------------------
# Prediction
# -------------------------------
Y_pred = model.predict(X_test)

# Convert back to original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# -------------------------------
# Evaluation
# -------------------------------
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

print("\nSample Predictions:")
print(Y_pred[:5])

```

## Output:

![Screenshot 2024-09-22 125825](https://github.com/user-attachments/assets/c72306bc-b687-48b6-b56c-af6dd2c98620)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
