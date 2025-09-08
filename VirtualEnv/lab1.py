# test_env.py
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Генерируем простые данные
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 точек, 1 признак
y = 3.5 * X.squeeze() + np.random.randn(100) * 2  # линейная зависимость с шумом

# Создаём модель
model = LinearRegression()
model.fit(X, y)

# Предсказания
y_pred = model.predict(X)

# Печатаем коэффициенты
print(f"Коэффициент: {model.coef_[0]:.2f}")
print(f"Смещение (intercept): {model.intercept_:.2f}")

# Визуализация
plt.scatter(X, y, color='blue', label='Данные')
plt.plot(X, y_pred, color='red', linewidth=2, label='Линейная регрессия')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Проверка виртуального окружения: Linear Regression')
plt.legend()
plt.show()
