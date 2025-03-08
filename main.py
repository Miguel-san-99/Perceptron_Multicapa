import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop, Adadelta


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

network = models.Sequential()

network.add(layers.Dense(300, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(200, activation='relu'))
network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

networkSGD = network
networkAdam = network
networkNadam = network
networkAdadelta = network
networkRMSprop = network

#network.compile(loss='categorical_crossentropy', optimizer=Adadelta(learning_rate=0.001), metrics=['accuracy', 'Precision'])
#history = network.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
networkAdam.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
historyAdam = networkAdam.fit(x_train, y_train, epochs=30)

networkAdadelta.compile(loss='categorical_crossentropy', optimizer=Adadelta(learning_rate=0.001), metrics=['accuracy'])
historyAdadelta = networkAdadelta.fit(x_train, y_train, epochs=30)

networkNadam.compile(loss='categorical_crossentropy', optimizer=Nadam(learning_rate=0.001), metrics=['accuracy'])
historyNadam = networkAdam.fit(x_train, y_train, epochs=30)

networkSGD.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
historySGD = networkAdadelta.fit(x_train, y_train, epochs=30)

networkRMSprop.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
historyRMSprop = networkAdam.fit(x_train, y_train, epochs=30)


# Convertir los historiales a DataFrames
df_adam = pd.DataFrame(historyAdam.history).add_prefix("Adam_")
df_adadelta = pd.DataFrame(historyAdadelta.history).add_prefix("Adadelta_")
df_rmsprop = pd.DataFrame(historyRMSprop.history).add_prefix("RMSprop_")
df_nadam = pd.DataFrame(historyNadam.history).add_prefix("Nadam_")
df_sgd = pd.DataFrame(historySGD.history).add_prefix("SGD_")

# Concatenar los DataFrames (unidos por índice, que representa las épocas)
df_combined = pd.concat([df_adam, df_adadelta, df_rmsprop, df_nadam, df_sgd], axis=1)

# Graficar
df_combined.plot(figsize=(12, 8))
plt.title("Comparación de Optimización entre Adam, Adadelta, RMSprop, Nadam y SGD")
plt.xlabel("Época")
plt.ylabel("Métrica")
plt.grid(True)
plt.show()

"""pd.DataFrame(historyAdam.history).plot(figsize=(10, 7))
pd.DataFrame(historyAdadelta.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.title("Comparacion de algoritmos")
plt.xlabel("epochs")
plt.show()
"""
"""predict = np.argmax(network.predict(x_test), axis= -1)
plt.imshow(x_test[243].reshape(28, 28))
# the label of the first number
plt.title(f"Digit: {predict[243]}") #aqui ponen el vector que quieren comparar con el numero que graficaron
plt.axis("off")
plt.show()"""

"""
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title("Ejemplo de imagen de prueba")
plt.show()
"""