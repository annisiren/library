import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# IMPORT DATASET
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING DTR MODEL ON WHOLE DATASET
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# PREDICT RESULT
y_pred = regressor.predict([[6.5]])

# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# EVALUATING MODEL PERFORMANCE
r2_score(y_test, y_pred)

# VISUALIZING DECISION TREE REGRESSION RESULTS (higher res)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
