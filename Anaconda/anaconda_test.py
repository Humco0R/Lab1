import gdown
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. Скачиваем файл с Google Drive ---
url = "https://drive.google.com/uc?id=1p2ogKsWW7fnWdHJFW5tWwqCEwG_exK-V"
output = "bottle.csv"
gdown.download(url, output, quiet=False)

# --- 2. Загружаем CSV ---
df = pd.read_csv(output)
print(df.head())
print(df.columns)

# Берём только нужные признаки
df_binary = df[['Salnty', 'T_degC']].copy()
df_binary.columns = ['Sal', 'Temp']

# Первичный график
sns.lmplot(x="Sal", y="Temp", data=df_binary, order=2, ci=None)
plt.title("Salinity vs Temperature (все данные)")
plt.show()

# Заполнение пропусков
df_binary.ffill(inplace=True)
df_binary.dropna(inplace=True)

# --- 3. Модель на всём датасете ---
X = df_binary['Sal'].to_numpy().reshape(-1, 1)
y = df_binary['Temp'].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)

print("=== Модель на всём датасете ===")
print(f"R²: {model_all.score(X_test, y_test):.4f}")

plt.scatter(X_test, y_test, color='blue', label="Test data")
plt.plot(X_test, y_pred_all, color='black', label="Prediction")
plt.title("Linear Regression (все данные)")
plt.xlabel("Sal")
plt.ylabel("Temp")
plt.legend()
plt.show()

# --- 4. Модель на первых 500 строках ---
df_binary500 = df_binary.iloc[:500].copy()

sns.lmplot(x="Sal", y="Temp", data=df_binary500, order=2, ci=None)
plt.title("Salinity vs Temperature (первые 500 строк)")
plt.show()

df_binary500.ffill(inplace=True)
df_binary500.dropna(inplace=True)

X_500 = df_binary500['Sal'].to_numpy().reshape(-1, 1)
y_500 = df_binary500['Temp'].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_500, y_500, test_size=0.25, random_state=42)

model_500 = LinearRegression()
model_500.fit(X_train, y_train)
y_pred_500 = model_500.predict(X_test)

print("\n=== Модель на первых 500 строках ===")
print(f"R²: {model_500.score(X_test, y_test):.4f}")

plt.scatter(X_test, y_test, color='blue', label="Test data")
plt.plot(X_test, y_pred_500, color='black', label="Prediction")
plt.title("Linear Regression (500 строк)")
plt.xlabel("Sal")
plt.ylabel("Temp")
plt.legend()
plt.show()

# --- 5. Метрики для второй модели ---
mae = mean_absolute_error(y_test, y_pred_500)
mse = mean_squared_error(y_test, y_pred_500)
rmse = np.sqrt(mse)  # работает во всех версиях sklearn

print("\n=== Ошибки (500 строк) ===")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
