import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Housing.csv')

# Display the first few rows
print(df.head())

# Encode categorical variables
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})
df['mainroad'] = df['mainroad'].map({'no': 0, 'yes': 1})
df['guestroom'] = df['guestroom'].map({'no': 0, 'yes': 1})
df['basement'] = df['basement'].map({'no': 0, 'yes': 1})
df['hotwaterheating'] = df['hotwaterheating'].map({'no': 0, 'yes': 1})
df['airconditioning'] = df['airconditioning'].map({'no': 0, 'yes': 1})
df['prefarea'] = df['prefarea'].map({'no': 0, 'yes': 1})

# Define features and target variable
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
            'basement', 'hotwaterheating', 'airconditioning', 'parking',
            'prefarea', 'furnishingstatus']
X = df[features]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
