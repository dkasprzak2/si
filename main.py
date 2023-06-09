import os
import numpy as np
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# Ścieżka do folderu zawierającego obrazy treningowe
train_data_dir = 'trening'
# Ścieżka do folderu zawierającego obrazy testowe
test_data_dir = 'test'

# Inicjalizacja parametrów
input_shape = (224, 224, 3)
num_classes = 3
batch_size = 32
epochs = 10

# Przygotowanie danych treningowych i testowych
train_data_generator = ImageDataGenerator(rescale=1.0/255.0)
test_data_generator = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Definicja modelu CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Uczenie modelu
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs)

# Testowanie modelu
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("Test accuracy:", test_acc)