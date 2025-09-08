import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Данные
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

# Модель
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"Коэффициент: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Визуализация
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.show()
