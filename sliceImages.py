# Importing all necessary libraries
import cv2
import os
import uuid

# LAST ONE RAN : FRAME 3
cam = cv2.VideoCapture("assets/rawFootage/frame3.mp4")

try:

    # creating a folder named data
    if not os.path.exists('assets/output'):
        os.makedirs('assets/output')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
fps = 30
seconds = 5
frameCondition = fps * seconds
while (True):
    ret, frame = cam.read()
    if not ret:
        break

    currentframe += 1

    if currentframe % frameCondition == 0:
        # if video is still left continue creating images
        imageId = str(uuid.uuid4())[0:7]
        name = './assets/output/' + imageId + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

cam.release()
cv2.destroyAllWindows()