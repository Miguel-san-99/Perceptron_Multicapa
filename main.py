import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



def prep_dataset(X, y, shape):
    x_prep = np.array(X).reshape(len(X), shape)
    x_prep = x_prep.astype("float32") / 255.0
    y_prep = to_categorical(np.array(y))
    return (x_prep, y_prep)

#Cargamos el dataset
data = pd.read_csv("MNIST/train.csv")
x_test = pd.read_csv("MNIST/test.csv").values

#Separamos etiquetas y pixeles
x_train = data.iloc[:, 1:].values
y_train = data.iloc[:, 0].values

#Comvertimos a a formato adecuado (28x28)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

#Escalamos los datos
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(np.array(y_train))
y_val = to_categorical(np.array(y_val))

print("x_train: ", x_train.shape)
print("x_val: ", x_val.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_val: ", y_val.shape)

network = models.Sequential()

network.add(layers.Dense(300, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(200, activation='relu'))
network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'Precision'])
history = network.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("epochs")
plt.show()

predict = np.argmax(network.predict(x_test), axis= -1)
print("Predict: ", predict.shape)
print("predict 240: ", predict[240])
"""
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title("Ejemplo de imagen de prueba")
plt.show()
"""