import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers


#Cargamos el dataset
data = pd.read_csv("MNIST/train.csv")
x_test = pd.read_csv("MNIST/test.csv").values

#Separamos etiquetas y pixeles
y_train = data.iloc[:, 0].values
x_train = data.iloc[:, 1:].values

#Comvertimos a a formato adecuado (28x28)
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

#Escalamos los datos
x_train = x_train / 255
x_test = x_test / 255

print("Shape: ", x_train.shape)
network = models.Sequential()

network.add(layers.Dense(300, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(200, activation='relu'))
network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'Presicion'])
history = network.fit(x_train, x_test, epochs=30)
"""
plt.imshow(x_test[0], cmap='gray')
plt.title("Ejemplo de imagen de prueba")
plt.show()
"""
