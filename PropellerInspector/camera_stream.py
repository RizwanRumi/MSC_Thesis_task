import numpy as np
import cv2

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

if not (video_capture_1.isOpened()):
    print("Could not open video device")

if not (video_capture_1.isOpened()):
    print("Could not open video device")


while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)

    if (ret1):
        # Display the resulting frame
        #cv2.imwrite("test.jpg", frame1)
        #img = cv2.imread("test.jpg")
        width = 640
        height = 640
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("./images/Edited.jpg", resized)
        print('Resized Dimensions : ', resized.shape)

        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()

