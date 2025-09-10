import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# --- Часть 1. Линейная регрессия на датасете diabetes ---
X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # Используем только один признак
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print("=== Diabetes dataset ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

# Train plot
ax[0].scatter(X_train, y_train, label="Train points")
ax[0].plot(X_train, lin_reg.predict(X_train), color="orange", linewidth=3, label="Model")
ax[0].set(title="Train set", xlabel="Feature", ylabel="Target")
ax[0].legend()

# Test plot
ax[1].scatter(X_test, y_test, label="Test points")
ax[1].plot(X_test, y_pred, color="orange", linewidth=3, label="Model")
ax[1].set(title="Test set", xlabel="Feature", ylabel="Target")
ax[1].legend()

fig.suptitle("Linear Regression on Diabetes dataset")

# --- Часть 2. Сравнение OLS и Ridge на игрушечных данных ---
X_train = np.c_[0.5, 1].T
y_train = [0.5, 1]
X_test = np.c_[0, 2].T

np.random.seed(0)

models = {"LinearRegression": LinearRegression(), "Ridge(alpha=0.1)": Ridge(alpha=0.1)}

for name, model in models.items():
    fig, ax = plt.subplots(figsize=(4, 3))

    # Несколько реализаций с шумом
    for _ in range(6):
        noisy_X = 0.1 * np.random.normal(size=(2, 1)) + X_train
        model.fit(noisy_X, y_train)
        ax.plot(X_test, model.predict(X_test), color="gray", alpha=0.6)

    # Основная модель без шума
    model.fit(X_train, y_train)
    ax.plot(X_test, model.predict(X_test), color="blue", linewidth=2, label="Fitted model")
    ax.scatter(X_train, y_train, color="red", marker="+", s=50, label="Train points")

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

plt.show()
