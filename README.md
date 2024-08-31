# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. A library for numerical computations.
4. Initialize theta with zeros
5. X.shape[1]
6. np.zeros(X.shape[1])
7. reshape(-1, 1)

8. 
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KARTHICKKUMAR R
RegisterNumber: 212223040087 
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to perform linear regression using gradient descent
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X1 for the intercept term
    X = np.c_[np.ones(len(X1)), X1]
    
    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)
    
    # Perform gradient descent
    for _ in range(num_iters):
        predictions = X.dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    
    return theta

# Load the data
data = pd.read_csv('/content/50_Startups.csv')

# Prepare the features and target
X = data.iloc[1:, :-2].values
y = (data.iloc[1:, -1].values).reshape(-1, 1)
X1 = X.astype(float)

# Scale the features and target
scaler = StandardScaler()
X1_Scaled = scaler.fit_transform(X1)
y_Scaled = scaler.fit_transform(y)

# Train the model
theta = linear_regression(X1_Scaled, y_Scaled)

# New data point for prediction
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)
new_Scaled = scaler.fit_transform(new_data)

# Make prediction
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1, 1)

# Inverse transform to get the prediction in the original scale
pre = scaler.inverse_transform(prediction)

# Output the prediction
print(f"Predicted value: {pre}")

```

## Output:

![img](https://github.com/user-attachments/assets/96752904-59c2-454c-a893-cb77dad739cf)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
