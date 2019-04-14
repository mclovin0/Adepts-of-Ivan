import numpy as np
import cv2
class EmotionClassifier:
    def __init__(self):
        self.face_emotion = cv2.face.FisherFaceRecognizer_create()
        self.path = 'models/emotion_classifier_model.xml'
        self.delay = 0
        self.init = True
        self.emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    def emotion_classify(self):
        image = cv2.imread("sample.jpg", 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (350, 350))
        self.face_emotion.read(self.path)
        if self.init or self.delay == 0:
            self.init = False
            emotion_prediction = self.face_emotion.predict(image)
            print(self.emotions[emotion_prediction[0]])


eml = EmotionClassifier()
eml.emotion_classify()
eml.emotion_classify()
