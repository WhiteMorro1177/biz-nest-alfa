import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Загружаем данные
# Предположим, что у нас есть DataFrame 'data' с координатами и целевой переменной (продажами).
# Пример данных: 'latitude', 'longitude', 'population_density', 'competition_index', 'sales'
data = pd.read_csv('data.csv')

# Предобработка данных
# Получаем X (признаки) и y (целевую переменную)
X = data[['latitude', 'longitude', 'population_density', 'competition_index']]
y = data['sales']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабируем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создаем модель нейронной сети
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Для регрессии, мы хотим предсказывать продажи
])

# Компилируем модель
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучаем модель
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Оцениваем модель
loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss}')

# Предсказание для новых координат
new_coordinates = np.array([[55.7558, 37.6173, 2000, 0.1]])  # Пример: [latitude, longitude, population_density, competition_index]
new_coordinates_scaled = scaler.transform(new_coordinates)
predicted_sales = model.predict(new_coordinates_scaled)

print(f'Predicted sales for new location: {predicted_sales[0][0]}')