import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Cargamos el dataset
dataset = pd.read_csv('CarPrice/carPriceDataset.csv')
X = dataset.drop('selling_price', axis=1)
Y = dataset['selling_price']

numerical = X.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner'], axis=1)
categorical = X.filter(['name', 'fuel', 'seller_type', 'transmission', 'owner'])

cat_numerical = pd.get_dummies(categorical, drop_first=True)
X = pd.concat([numerical, cat_numerical], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LinearRegression()
regressor = lr.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R²:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)

plt.title("Valores Reales vs Predicciones")
plt.xlabel("Valores Reales")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.grid(True)
plt.show()

errors = y_pred - y_test

# Creamos el histograma con la curva de densidad
plt.figure(figsize=(10,6))
sns.histplot(errors, bins=80, kde=True)

plt.title("Distribución de los Errores de Predicción")
plt.xlabel("Error de Predicción (Predicción - Valor Real)")
plt.ylabel("Frecuencia")
plt.show()