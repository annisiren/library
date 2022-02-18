import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# IMPORT DATA
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING LINEAR REGRESSION MODEL ON WHOLE DATASET
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# TRAINING POLYNOMIAL REGRESSION MODEL ON WHOLE DATASET
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# VISUALIZING LINEAR REGRESSION RESULTS
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# VISUALIZING POLYNOMIAL REGRESSION RESULTS
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# VISUALIZING POLYNOMIAL REGRESSION RESULTS (higher res, smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# PREDICTING NEW RESULT WITH LINEAR REGRESSION
lin_reg.predict([[6.5]])

# PREDICTING NEW RESULT WITH POLYNOMIAL REGRESSION
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
