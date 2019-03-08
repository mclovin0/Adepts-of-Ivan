import numpy as np
from keras.models import load_model
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_emotion(image):
    emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_classifier = load_model(emotion_model_path)
    image = preprocess_input(image, True)
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, -1)
    emotion_prediction = emotion_classifier.predict(image)
    emotion_probability_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_probability_arg]
    return emotion_text

