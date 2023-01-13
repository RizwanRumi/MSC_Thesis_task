"""
This code is only for camera testing purpose
"""

import numpy as np
import cv2

#video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(0)
#video_capture_2 = cv2.VideoCapture(2)

video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

#video_capture_2.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
#video_capture_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

if not (video_capture_1.isOpened()):
    print("Could not open video device 1")

#if not (video_capture_2.isOpened()):
#    print("Could not open video device 2")


while True:
    # Capture frame-by-frame
    #ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    #ret2, frame2 = video_capture_2.read()

    #if (ret0):
        # Display the resulting frame
    #    cv2.imshow('Cam 0', frame0)

    if (ret1):
        # Display the resulting frame
        width = 640
        height = 640
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("./images/Camera_1.jpg", resized)
        print('Resized Dimensions : ', resized.shape)

        cv2.imshow('Cam 1', frame1)
    """
    
    if (ret2):
        # Display the resulting frame
        width = 640
        height = 640
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("./images/Camera_2.jpg", resized)
        print('Resized Dimensions : ', resized.shape)

        cv2.imshow('Cam 2', frame2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """

    # key: 'ESC'
    key = cv2.waitKey(20)
    if key == 27:
        break

# When everything is done, release the capture
#video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()

