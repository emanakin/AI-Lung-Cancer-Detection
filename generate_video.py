import cv2
import numpy as np
import os
import tensorflow as tf

# Load the trained models
custom_cnn = tf.keras.models.load_model('models/custom_cnn.keras')
vgg16 = tf.keras.models.load_model('models/vgg16.keras')
resnet50 = tf.keras.models.load_model('models/resnet50.keras')
inceptionv3 = tf.keras.models.load_model('models/inceptionv3.keras')

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_expanded = np.expand_dims(img, axis=[0, -1])
    img_expanded_rgb = np.repeat(img_expanded, 3, axis=-1)
    return img_expanded, img_expanded_rgb

def predict(image, model):
    img, img_rgb = preprocess_image(image)
    pred = model.predict(img_rgb)[0][0] if len(model.input_shape) == 4 else model.predict(img)[0][0]
    label = "Cancer" if pred > 0.5 else "No Cancer"
    return label, pred

def generate_video(output_path, model, model_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, 1, (224, 224))

    # Assuming your images are in the 'data/test_images/' directory
    image_files = [f for f in os.listdir('data/test_images/') if f.endswith('.png')]
    
    for image_file in image_files:
        image = cv2.imread(os.path.join('data/test_images/', image_file))
        label, confidence = predict(image, model)
        
        # Overlay text on the image
        text = f"{model_name}: {label} ({confidence:.2f})"
        cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        video_writer.write(image)
    
    video_writer.release()

# Generate videos for each model
generate_video('output_custom_cnn.avi', custom_cnn, 'Custom CNN')
generate_video('output_vgg16.avi', vgg16, 'VGG16')
generate_video('output_resnet50.avi', resnet50, 'ResNet50')
generate_video('output_inceptionv3.avi', inceptionv3, 'InceptionV3')

# Generate video for ensemble model
def predict_ensemble(image):
    img, img_rgb = preprocess_image(image)
    custom_pred = custom_cnn.predict(img)[0][0]
    vgg16_pred = vgg16.predict(img_rgb)[0][0]
    resnet50_pred = resnet50.predict(img_rgb)[0][0]
    inceptionv3_pred = inceptionv3.predict(img_rgb)[0][0]
    
    ensemble_pred = np.mean([custom_pred, vgg16_pred, resnet50_pred, inceptionv3_pred])
    label = "Cancer" if ensemble_pred > 0.5 else "No Cancer"
    return label, ensemble_pred

def generate_ensemble_video(output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, 1, (224, 224))

    image_files = [f for f in os.listdir('data/test_images/') if f.endswith('.png')]
    
    for image_file in image_files:
        image = cv2.imread(os.path.join('data/test_images/', image_file))
        label, confidence = predict_ensemble(image)
        
        text = f"Ensemble: {label} ({confidence:.2f})"
        cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        video_writer.write(image)
    
    video_writer.release()

generate_ensemble_video('output_ensemble.avi')
