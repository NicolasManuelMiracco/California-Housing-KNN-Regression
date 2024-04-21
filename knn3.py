from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar el conjunto de datos
data = fetch_california_housing()
X = data.data
y = data.target

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def knn_regression(X_train, y_train, test_instance, k=3):
    # Calcular las distancias entre la instancia de prueba y todas las instancias de entrenamiento
    distances = np.sqrt(((X_train - test_instance) ** 2).sum(axis=1))
    
    # Obtener los índices de las k menores distancias
    k_indices = distances.argsort()[:k]
    
    # Promediar los valores objetivo de los k vecinos más cercanos
    prediction = y_train[k_indices].mean()
    
    return prediction

# Predecir para una instancia única en el conjunto de prueba para demostración
test_instance = X_test[0]
predicted_price = knn_regression(X_train, y_train, test_instance, k=3)
print(f"Predicted Price: ${predicted_price:.2f}")
print(f"Actual Price: ${y_test[0]:.2f}")
