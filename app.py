import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential, load_model
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Activation

with st.container():
    st.title("Trabajo Practico IA Deployement")
    st.write("Esta es la pagina de pruebas para nuestro IA")

class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_path = st.file_uploader("Choose a h5 file", type="hdf5")

path_best_model = 'best_model.hdf5'

loaded_model = model.load_weights(model_path)

image_file = st.file_uploader("Upload images for vegetable classification", type=['png','jpeg'])

def predict(image):
    img = load_img(image, target_size = (256,256))

    # La convierto a un numpy array
    img = img_to_array(img)

    # Le sumo una dimesion, cuyo valor es cero. 
    # De (256, 256, 3) a (256, 256, 3, 0)
    # Esto porque la funcion flow para cargar con el iterator la imagen necesita un array de 4 dimensiones como input
    img = expand_dims(img, 0)

    # ImageDataGenerator brillo
    datagen = ImageDataGenerator(rescale=1./255)

    test_img = datagen.flow(img)

    predictions = loaded_model.predict(test_img)

    number_class_pred = np.argmax(predictions)

    prediction = class_names[number_class_pred]

    return prediction

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image)

detect = st.button("Classify")

if detect:
    prediction = predict(image_file)
    st.write(prediction)
