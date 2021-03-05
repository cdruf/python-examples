import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from pathlib import Path

# %%

print(Path.cwd())
df = pd.read_csv("dat/Housing.csv")
print(df.head())

X = df['lotsize'].to_numpy().reshape(-1, 1)
Y = df['price'].to_numpy().reshape(-1, 1)

# Split dat into training and testing sets
X_train = X[:-250]
Y_train = Y[:-250]
X_test = X[-250:]
Y_test = Y[-250:]

# Plot outputs
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Lot size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

plt.show()

# Regression
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)
plt.plot(X_test, lm.predict(X_test), color='red', linewidth=4)
