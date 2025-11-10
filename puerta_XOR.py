import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Dataset de entrenamiento para puerta XOR
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y_train = np.array([[0], [1], [1], [0]], dtype="float32")

# Modelo MLP
model = keras.Sequential()
model.add(layers.Dense(2, input_dim=2, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuración del modelo
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
    metrics=['accuracy']
)

# Entrenamiento
fit_history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=0)

# Gráfica de pérdida
loss_curve = fit_history.history['loss']
plt.plot(loss_curve, label='Pérdida')
plt.legend(loc='lower left')
plt.title('Resultado del Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()

# Recuperamos pesos y sesgos de las capas
weights_HL, biases_HL = model.layers[0].get_weights()
weights_OL, biases_OL = model.layers[1].get_weights()

print("Pesos capa oculta:\n", weights_HL)
print("Bias capa oculta:\n", biases_HL)
print("Pesos capa de salida:\n", weights_OL)
print("Bias capa de salida:\n", biases_OL)

# Predicciones
prediccion = model.predict(x_train)
print("Predicciones:\n", prediccion)
print("Entradas:\n", x_train)
print("Salidas esperadas:\n", y_train)
