# The following program serves the purpose of facial detection so that a robotic webcam device can track the subject of
# the video feed in the center of the frame. Since the device is going to be trained to only track the author, it will
# be called the "NarCam" (short for Narcissistic + Cameron/Camera)

# The Program will start by creating training a model to specifically recognize the face of the author (Cameron Yenche)
# by applying transfer learning to a model pretrained to recognized faces. This process consists of manually labeling
# data and utilizing it as the top layer of the custom-model. The program will then utilize openCV packages to identify
# the face in real time, bound it, and output the center coordinates of the bounding box

# Copyright: Cameron Yenche

# We will begin the program by setting up path shortcuts:

#  WORKSPACE_PATH = "Tensorflow/workspace"
# SCRIPTS_PATH = "Tensorflow/scripts"
# API_MODEL_PATH = "Tensorflow/models"
# ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
# IMAGE_PATH = WORKSPACE_PATH + "/images"
# MODEL_PATH = WORKSPACE_PATH + "/models"
# PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"
# CONFIG_PATH = MODEL_PATH + "/my_ssd_mobnet/pipeline.config"
# CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Capture local video feed from webcam


def detect_face_cascades(img_frame):
    # Import the frontal face recognition model
    face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_img = img_frame.copy()
    face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_rect = face_cascade_front.detectMultiScale(face_img_gray, scaleFactor=1.2, minNeighbors=5)

    rect_coord = []  # Initialize Rectangle coordinate array

    for (x, y, w, h) in face_rect:
        rect_coord = (x, y)
        print(rect_coord)
        face_img = cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img, rect_coord


def detect_face():
    while True:
        ret, frame = cap.read()

        face_img = detect_face_cascades(frame)

        cv2.imshow('Face Detection Window', face_img[0])  # Display Video Stream

        if cv2.waitKey(1) == ord('q'):  # Exit the video capture loop if q is pressed on the keyboard
            break

    cap.release()
    cv2.destroyAllWindows()


detect_face()
