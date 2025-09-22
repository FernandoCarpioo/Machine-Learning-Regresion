import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

X  = df.drop('target', axis=1)
Y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=137)

knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
regressor = knn.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R²:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)

plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.show()

errors = y_pred - y_test

# Creamos el histograma con la curva de densidad
plt.figure(figsize=(10,6))
sns.histplot(errors, bins=80, kde=True)

plt.title("Distribución de los Errores de Predicción")
plt.xlabel("Error de Predicción (Predicción - Valor Real)")
plt.ylabel("Frecuencia")
plt.show()