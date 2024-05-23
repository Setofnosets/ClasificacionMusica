import datetime
import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras import regularizers
import glob
import random
import shutil

espectrogramas = "Data/images_original"
separacion = ["Data/train", "Data/test", "Data/validation"]
train_dir = "Data/train"
test_dir = "Data/test"
validation_dir = "Data/validation"

for s in separacion:
    if not os.path.isdir(s):
        os.mkdir(s)

generos = list(os.listdir(espectrogramas))
# Orden alfabetico
generos.sort()

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(432, 288)
)

# Capas: Entrada: 64
# Capa convolucional: 128
# Capa de pooling: 2
# Capa convolucional: 256
# Capa de pooling: 2
# Capa convolucional: 256
# Capa de pooling: 2
# Capa Densa: 512
# Capa Densa: 256
# Capa de salida: 12
tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="MyLayers", name="Modelo")
class Modelo(models.Model):
    def __init__(self, **kwargs):
        super(Modelo, self).__init__()
        # Entrada
        self.input_layer = layers.Conv2D(64, (3, 3), activation='relu', input_shape=(432, 288, 3),
                                         kernel_regularizer=regularizers.l2(0.1))
        # Batch Normalization
        self.batch_norm = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout = layers.Dropout(0.5)
        # Convolucional y pooling
        self.conv1 = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1))
        self.batch_norm1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout1 = layers.Dropout(0.8)
        # Convolucional y pooling
        self.conv2 = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1))
        self.batch_norm2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout2 = layers.Dropout(0.8)
        # Convolucional y pooling
        self.conv3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1))
        self.batch_norm3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout3 = layers.Dropout(0.8)
        # Flatten
        self.flatten = layers.Flatten()
        # Densa
        self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.1))
        self.batch_norm4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.8)
        # Densa
        self.dense2 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1))
        self.batch_norm5 = layers.BatchNormalization()
        self.dropout5 = layers.Dropout(0.8)
        # Salida
        self.output_layer = layers.Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(0.1))

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.output_layer(x)
        return x

model = Modelo()

# Cargar el modelo
model = tf.keras.models.load_model('modelNewDataset.keras', custom_objects={'Modelo': Modelo})

# Metricas

y_pred = np.array([])
y_true = np.array([])
for x, y in test_dataset:
    y_pred = np.concatenate([y_pred, np.argmax(model.predict(x), axis=1)])
    y_true = np.concatenate([y_true, np.argmax(y, axis=1)])

# Matriz de confusion

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=generos, yticklabels=generos)
plt.xlabel('Predecido')
plt.ylabel('Verdadero')
plt.show()

# Precision, recall y f1-score

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=generos))

# Predecir
import numpy as np
from tensorflow.keras.preprocessing import image

print("Prueba general")
model.evaluate(test_dataset)

# Recuperar un elemento aleatorio de cada genero
print("Prueba Aleatorea")
for g in generos:
    files = glob.glob(f"{test_dir}/{g}/*.png")
    img_path = random.choice(files)
    img = image.load_img(img_path, target_size=(432, 288))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)

    print("Esta imagen probablemente pertenece a {} con una confianza de {:.2f} por ciento. El verdadero valor es: {}."
          .format(generos[np.argmax(score)], 100 * np.max(score), g))

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        "Esta imagen probablemente pertenece a {} con una confianza de {:.2f} por ciento. El verdadero valor es: {}."
        .format(generos[np.argmax(score)], 100 * np.max(score), g)
    )
    plt.show()

# Prueba individual:
print("Prueba Individual")
img_path = 'Thriller.png'
img = image.load_img(img_path, target_size=(432, 288))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Esta imagen pertenece a {} con una confianza de {:.2f} por ciento."
    .format(generos[np.argmax(score)], 100 * np.max(score))
)

