import cv2
import os
from imutils.video import VideoStream

def capture_face(user_id, save_directory):
    cam = VideoStream(src=0).start()

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("\n [INFO] Initializing face capture. Look at the camera and wait...")

    count = 0
    while True:
        img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            face_path = os.path.join(save_directory, f"User_{user_id}_{count}.jpg")
            cv2.imwrite(face_path, gray[y:y + h, x:x + w])
            cv2.imshow('Capturing Face', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27 or count >= 5:  # Press 'ESC' to exit or capture 5 face samples
            break

    print("\n [INFO] Exiting face capture. Cleaning up...")
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_id = input('\nEnter user ID and press <return>: ')
    images_directory = 'images'
    os.makedirs(images_directory, exist_ok=True)
    
    capture_face(user_id, images_directory)






