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

# Generacion de train, test y validation
"""
# Vaciar los directorios de train, test y validation
for s in separacion:
    for g in generos:
        if os.path.isdir(f"{s}/{g}"):
            shutil.rmtree(f"{s}/{g}")

for g in generos:
  # Dividir imagenes en train, test y validation
  src_files = []
  for file in glob.glob(f"{espectrogramas}/{g}/*.png", recursive=True):
    src_files.append(file)
  random.shuffle(src_files)
  # Test files: 10%, Validation files: 10%, Train files: 80%
  train_files, validation_files, test_files = np.split(src_files, [int(.8*len(src_files)), int(.9*len(src_files))])

  for f in separacion:
    if not os.path.isdir(f"{f}/{g}"):
      os.mkdir(f"{f}/{g}")

  for f in train_files:
      if not os.path.isfile(f"{train_dir}/{g}/{os.path.basename(f)}"):
        os.rename(f, f"{train_dir}/{g}/{os.path.basename(f)}")
  for f in test_files:
      if not os.path.isfile(f"{test_dir}/{g}/{os.path.basename(f)}"):
        os.rename(f, f"{test_dir}/{g}/{os.path.basename(f)}")
  for f in validation_files:
      if not os.path.isfile(f"{validation_dir}/{g}/{os.path.basename(f)}"):
        os.rename(f, f"{validation_dir}/{g}/{os.path.basename(f)}")
"""
# Crear el modelo

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(432, 288)
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(432, 288)
)

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


# Entrenamiento
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, weight_decay=0.001, nesterov=True),
                loss='categorical_crossentropy',
                metrics=['accuracy']
              )

log_dir = "logs/fit/L2_kernel_0.0001_Bigger_SGD_callback_decay" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=tensorboard_callback)

# Guardar el modelo
model.save('modelNewDataset.keras')