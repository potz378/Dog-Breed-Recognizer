import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

IMAGE_SIZE = (450, 450)

'''
 Dataset folder should look like
 | datasets
    | breed_1
        |image_1
        |image_2
    | breed_2
        |image_1
        |image_2
 | breed_recognition.py
'''

#Label Encoding
def label_encoder(y):
    categories = list(set(y))
    categories.sort()
    new_y = [categories.index(y) for y in y]
    return new_y

# Load the dataset
path_to_dataset = Path.cwd().joinpath('datasets')
breeds = list(path_to_dataset.glob('*'))
images = []
for breed in breeds:
    for image in list(breed.glob('*')):
        images.append(image)

# X
X = [img_to_array(load_img(image)) for image in images]
X = [tf.image.resize(array, IMAGE_SIZE, method=ResizeMethod.BILINEAR) for array in X]
X = np.array(X, dtype=np.float32)

# Preprocessing
X = X / 255.0

# Y
y = [image.parts[-2] for image in images]
y = label_encoder(y)
y = to_categorical(y)  # One hot encoding
y = np.array(y, dtype=np.float32)
output_categories = len(np.unique(y))
# Note from the future: This is dumb VVV
y = y[:, :-1]  # Avoiding a relation between independent variables

# Training
data_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, height_shift_range=0.1, width_shift_range=0.1, brightness_range=[0.3, 1.5], rotation_range=45, zoom_range=0.3)
train_gen = data_gen.flow(X, y, batch_size=5, seed=420)


##Note from github special editor: OH SH*T I DIDN'T NOTICED THESE !!?!
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(output_categories, activation='relu'),
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])

result = model.fit(train_gen, epochs=10, steps_per_epoch=X.shape[0] // 5)
model.save('model.h5')
