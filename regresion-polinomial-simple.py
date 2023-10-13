import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar tus datos en un DataFrame
data = pd.read_csv('polynomial-regression.csv')

# Obtener las características (X) y la variable objetivo (y)
X = data['araba_max_hiz'].values  # Cambiar a 'araba_max_hiz' si se desea usar este campo
y = data['araba_fiyat'].values

# Grado del polinomio
degree = 2  # Puedes ajustar el grado del polinomio

# Inicializar listas para almacenar los errores de validación cruzada
mse_scores = []

# Realizar validación cruzada "leave-one-out"
for i in range(len(X)):
    # Dejar un dato afuera para validar
    X_train = np.delete(X, i)
    y_train = np.delete(y, i)
    X_test = X[i]
    y_test = y[i]

    # Aplicar transformación polinomial a las características
    X_train_poly = np.column_stack([X_train ** i for i in range(1, degree + 1)])
    X_test_poly = np.array([X_test ** i for i in range(1, degree + 1)])

    # Crear y entrenar el modelo de regresión lineal
    coef = np.polyfit(X_train, y_train, degree)

    # Evaluar el modelo en el dato de prueba
    y_pred = np.polyval(coef, X_test_poly)

    # Calcular el error cuadrado medio (MSE)
    mse = ((y_pred - y_test) ** 2)
    mse_scores.append(mse)

# Calcular el error cuadrado medio promedio de la validación cruzada
average_mse = np.mean(mse_scores)

print(f'Error cuadrado medio promedio: {average_mse}')

# Visualización del error cuadrado medio promedio
plt.figure()
plt.plot(range(1, len(X) + 1), mse_scores, marker='o')
plt.xlabel('Iteración de Validación Cruzada')
plt.ylabel('Error Cuadrado Medio')
plt.title('Error Cuadrado Medio en Validación Cruzada Leave-One-Out')
plt.grid()
plt.show()