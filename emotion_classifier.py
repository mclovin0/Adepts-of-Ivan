import numpy as np
from keras.models import load_model
class EmotionClassifier:
    def __init__(self, image):
        self.emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
        self.emotion_classifier = load_model('./models/emotion_model.hdf5')
        self.image = image
        self.emotion_prediction = ''
        self.emotion_probability_arg = ''
        self.emotion_text = ''
    def preprocess_input(self, v2=True):
        self.image = self.image.astype('float32')
        self.image = self.image / 255.0
        if v2:
            self.image = self.image - 0.5
            self.image = self.image * 2.0
        self.image = np.expand_dims(self.image, 0)
        self.image = np.expand_dims(self.image, -1)
        return self.image
    def get_emotion(self):
        self.emotion_prediction = self.emotion_classifier.predict(self.preprocess_input(self.image))
        self.emotion_probability_arg = np.argmax(self.emotion_prediction)
        self.emotion_text = self.emotion_labels[self.emotion_probability_arg]
        return self.emotion_text
