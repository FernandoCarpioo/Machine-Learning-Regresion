from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


diabetes = load_diabetes(as_frame=True)
df = diabetes.data
df['target'] = diabetes.target  # Agregamos la variable objetivo


df.plot(kind='scatter', x='bmi', y='target')
plt.title("Relación IMC vs Progresión Diabetes")
plt.show()


X = df.drop(columns=['target'])
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


Lr = LinearRegression()
Lr.fit(X_train, y_train)


prediccion = Lr.predict(X_test)


plt.scatter(y_test, prediccion, alpha=0.7, label='Valores reales', color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], label='Regresión lineal', color='green')
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.legend()
plt.show()


r = Lr.score(X_test, y_test)
mae = mean_absolute_error(y_test, prediccion)
mse = mean_squared_error(y_test, prediccion)

print(f"El valor de R^2 para este modelo es: {r}")
print(f"Error Absoluto Medio (MAE): {mae}")
print(f"Error Cuadrático Medio (MSE): {mse}")
