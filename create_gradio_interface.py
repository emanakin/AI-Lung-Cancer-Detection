import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the trained models
custom_cnn = tf.keras.models.load_model('models/cnn/custom_cnn.keras')
vgg16 = tf.keras.models.load_model('models/pretrained/vgg16.keras')
resnet50 = tf.keras.models.load_model('models/pretrained/resnet50.keras')
inceptionv3 = tf.keras.models.load_model('models/pretrained/inceptionv3.keras')

# Function to preprocess image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_expanded = np.expand_dims(img, axis=[0, -1])
    img_expanded_rgb = np.repeat(img_expanded, 3, axis=-1)
    return img_expanded, img_expanded_rgb

# Functions to make predictions with each model
def predict_custom_cnn(image):
    img, _ = preprocess_image(image)
    pred = custom_cnn.predict(img)[0][0]
    label = "Cancer" if pred > 0.5 else "No Cancer"
    confidence = float(pred)
    return label, confidence

def predict_vgg16(image):
    _, img_rgb = preprocess_image(image)
    pred = vgg16.predict(img_rgb)[0][0]
    label = "Cancer" if pred > 0.5 else "No Cancer"
    confidence = float(pred)
    return label, confidence

def predict_resnet50(image):
    _, img_rgb = preprocess_image(image)
    pred = resnet50.predict(img_rgb)[0][0]
    label = "Cancer" if pred > 0.5 else "No Cancer"
    confidence = float(pred)
    return label, confidence

def predict_inceptionv3(image):
    _, img_rgb = preprocess_image(image)
    pred = inceptionv3.predict(img_rgb)[0][0]
    label = "Cancer" if pred > 0.5 else "No Cancer"
    confidence = float(pred)
    return label, confidence

def predict_ensemble(image):
    img, img_rgb = preprocess_image(image)
    custom_pred = custom_cnn.predict(img)[0][0]
    vgg16_pred = vgg16.predict(img_rgb)[0][0]
    resnet50_pred = resnet50.predict(img_rgb)[0][0]
    inceptionv3_pred = inceptionv3.predict(img_rgb)[0][0]
    
    ensemble_pred = np.mean([custom_pred, vgg16_pred, resnet50_pred, inceptionv3_pred])
    label = "Cancer" if ensemble_pred > 0.5 else "No Cancer"
    confidence = float(ensemble_pred)
    return label, confidence

custom_interface = gr.Interface(fn=predict_custom_cnn, inputs="image", outputs=["label", "number"], title="Custom CNN")
vgg16_interface = gr.Interface(fn=predict_vgg16, inputs="image", outputs=["label", "number"], title="VGG16")
resnet50_interface = gr.Interface(fn=predict_resnet50, inputs="image", outputs=["label", "number"], title="ResNet50")
inceptionv3_interface = gr.Interface(fn=predict_inceptionv3, inputs="image", outputs=["label", "number"], title="InceptionV3")
ensemble_interface = gr.Interface(fn=predict_ensemble, inputs="image", outputs=["label", "number"], title="Ensemble Model")

gr.TabbedInterface([custom_interface, vgg16_interface, resnet50_interface, inceptionv3_interface, ensemble_interface], ["Custom CNN", "VGG16", "ResNet50", "InceptionV3", "Ensemble"]).launch()


# interface = gr.Interface(fn=predict, inputs="image", outputs=["label", "json"])
# interface.launch()

