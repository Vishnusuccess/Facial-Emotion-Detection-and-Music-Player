import cv2
import os
import numpy as np
import random
import pygame
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

pygame.mixer.init()

# Define the emotion-to-song recommendation mapping
emotion_folders = {
    "Happy": "/Users/vishnu/Desktop/Facial Emotion detection and Music Player/music",
    "Surprise": "/Users/vishnu/Desktop/Facial Emotion detection and Music Player/music",
    # Add more emotions and corresponding music paths as needed
    "Sad": "music/sad_audio1.mpeg",
}

# Load the pre-trained facial emotion detection model
model = load_model('fer_weights.hdf5')

# Load the pre-trained facial emotion detection model (such as OpenCV or CNN-based models)
def detect_emotion(image):
    # Implement your facial emotion detection algorithm here
    # Perform emotion classification and return the detected emotion
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    emotion_label = "Neutral"  # Initialize the variable with a default value

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 64, 64, 1)

        emotion_preds = model.predict(roi)[0]
        emotion_label = class_labels[emotion_preds.argmax()]  # Find the label
        emotion_label_position = (x, y)
        cv2.putText(image, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', image)
    return emotion_label

# Initialize the webcam or video feed
cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate video source index if using multiple cameras

start_time = time.time()
song_playing = False
while time.time() - start_time < 5:
    ret, frame = cap.read()
    
    # Perform facial emotion detection on the current frame
    emotion = detect_emotion(frame)
    
    # Display the frame with emotion detection (optional)
    cv2.imshow("Emotion Detection", frame)
    
    # Wait for a key press to end the loop (optional)
    if cv2.waitKey(1) != -1:
        break

# Set the flag to indicate that the 5-second emotion detection is over
song_playing = True

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Play the song only if the flag is set
if song_playing:
    # Retrieve the corresponding folder path for the detected emotion
    emotion_folder = emotion_folders.get(emotion)

    if emotion_folder:
        # Get the list of music files in the emotion folder
        music_files = os.listdir(emotion_folder)

        if music_files:
            # Select a random music file from the emotion folder
            random_music = random.choice(music_files)

            # Play the randomly selected music file
            music_path = os.path.join(emotion_folder, random_music)
            pygame.mixer.music.load(music_path)

            # Play the music file
            pygame.mixer.music.play()

# Wait for the song to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)







