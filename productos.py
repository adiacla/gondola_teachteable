# streamlit_audio_recorder y whisper by Alfredo Diaz - version April 2024
#python -m venv env
#cd D:\smart\env\Scripts\
#.\activate 
#cd d:\mango
#pip install tensorflow==2.12
#pip install numpy
#pip install streamlit
#pip install pillow
##pip install tensorflow==2.12

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf # TensorFlow is required for Keras to work
from PIL import Image
import numpy as np
import cv2

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Reconocimiento de productos de Retail",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./converted_keras/keras_model.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()
    
    
def prediction_cls(prediction): # predecir la clase de las imágenes en función de los resultados del modelo
    for key, clss in class_names.items(): # crear un diccionario de las clases de salida
        if np.argmax(prediction)==clss: # Verifica la clase
            return key

with st.sidebar:
        st.image('cart.jpg')
        st.title("REconocimiento de imagen")
        st.subheader("Reconocimiento de imagen de produtos de retail")

st.image('logo.png')
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de poductos de góndola
         """
         )


def import_and_predict(image_data, model, class_names):
    # Redimensionar la imagen a 224x224
    image_data = cv2.resize(image_data, (224, 224))
    
    # Convertir la imagen a un array numpy y cambiar su forma al shape requerido por el modelo
    image = np.asarray(image_data, dtype=np.float32).reshape(1, 224, 224, 3)
    
    # Normalizar la imagen
    image = (image / 127.5) - 1
    
    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

class_names = open("./converted_keras/labels.txt", "r").readlines()

# Mostrar la vista de webcam

# Mostrar la vista previa de la webcam y capturar la imagen
# Mostrar la vista previa de la webcam y capturar la imagen
# Mostrar la vista previa de la webcam y capturar la imagen
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("No se pudo abrir la cámara.")
else:
    st.write("Presiona 'Detener' para detener la captura.")

    stop_button_key = "stop_button_" + str(hash("stop_button"))  # Generar una clave única para el botón
    stop_button = st.button("Detener", key=stop_button_key)

    while not stop_button:
        ret, frame = cap.read()
        if ret:
            # Convertir el fotograma capturado a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Realizar la predicción
            class_name, confidence_score = import_and_predict(frame_rgb, model, class_names)

            # Mostrar el video en tiempo real y la predicción
            st.video(frame_rgb, format="RGB")
            st.text(f"Producto: {class_name}")
            st.text(f"Puntuación de confianza: {confidence_score:.2f}")

            # Actualizar el botón de detener
            stop_button = st.button("Detener", key=stop_button_key)

    else:
        st.error("Se detuvo la captura de video.")

# Liberar los recursos de la webcam
cap.release()