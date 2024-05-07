
#pip install tensorflow==2.12

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./converted_keras/keras_Model.h5", compile=False)


# Load the labels
class_names = open("./converted_keras/labels.txt", "r").readlines()


# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)


# Create a window with a larger size
cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Image", 800, 600)  # You can adjust the size as needed

while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_input = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_input = (image_input / 127.5) - 1

    # Predict with the model
    prediction = model.predict(image_input)
    confidence_score = prediction.max()
    class_index = np.argmax(prediction)
    class_name = class_names[class_index][2:].strip()

    # Check if confidence_score is less than 0.5
    if confidence_score < 0.5:
        class_name = "Presente un producto"
    else:
        class_name = class_name

    # Draw class_name and confidence_score on the image
    text = f"{class_name}: {confidence_score:.2f}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()