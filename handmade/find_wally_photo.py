# This script takes a photo via the webcam and saves it for further processing.

import os
import cv2
import sys

sys.path.append("..")
sys.path.insert(0, '/Users/taras/Projects/uni/wallyCV-model/handmade/tensorflow/research')

video = cv2.VideoCapture(0)
# Set the highest resolution you can squeeze out from your webcam
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
print("Frame resolution set to: (" + str(video.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")


def screenshot():
    global cam
    cv2.imshow('img1', cam.read()[1])


while True:
    ret, frame = video.read()
    cv2.imshow('AVPRG', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('/Users/taras/Desktop/webcam_AVPRG_01.jpg', frame)
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('n'):
        screenshot()

video.release()
