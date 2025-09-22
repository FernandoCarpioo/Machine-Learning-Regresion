import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#cargamos el dataset e imprimimos las primieras 5 filas
dataset = pd.read_csv('carPrice/carPriceDataset.csv')
#Eliminamos la etiqueta que se va a predecir y la asignamos a y
X = dataset.drop('selling_price', axis=1)
Y = dataset['selling_price']

#quitamos variables categoricas y las guardamos en una variable
numerical = X.drop(['name','fuel','seller_type','transmission','owner'], axis=1)
categorical = X.filter(['name','fuel','seller_type','transmission','owner'])

#convertimos variables categoricas a numericas y lo concatenamos en X
cat_numerical = pd.get_dummies(categorical,drop_first=True)
X = pd.concat([numerical, cat_numerical], axis=1)

#separamos datos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

#Escalamos o normalizamos los datos de los sets de entrenamiento y test
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

rf_reg = RandomForestRegressor(random_state=40, n_estimators=200)
rf_reg.fit(x_train, y_train)
regressor = rf_reg.fit(x_train, y_train)
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