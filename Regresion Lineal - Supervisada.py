from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#Visualizacion del database iris
iris = load_iris(as_frame=True)
df =iris.data
df.plot(kind='scatter',x='petal length (cm)',y='petal width (cm)')
plt.show()



X = df.drop(columns=['petal length (cm)']) 
y = df['petal length (cm)']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


Lr = LinearRegression()
Lr.fit(X_train, y_train)


prediccion = Lr.predict(X_test)


plt.scatter(y_test, prediccion, alpha=0.7, label= 'valores reales', color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],label='Regresion lineal', color='green')
plt.legend()
plt.show()

print("El valor de nuestro R^2 para este modelo, es el siguiente.\n")
Lr.score(X_test, y_test)



