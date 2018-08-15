# https://realpython.com/face-recognition-with-python/

import cv2
import sys
import glob

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Get user supplied values
#imagePath = "AF20.jpg"

def go(photo):
    image = cv2.imread(photo + ".jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print('width of',w)
        print('height of',h)
        print('x location',x)
        print('y location',y)
    cv2.imshow("Faces found", image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    
def loop():
    files = glob.glob ("C:/Users/Miles/Desktop/python/faces/SCUT-FBP5500_v2/Images/*.jpg")
    for myFile in files:
        print(myFile)
        image = cv2.imread(myFile)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print('width of',w)
            print('height of',h)
            print('x location',x)
            print('y location',y)
        cv2.imshow("Faces found", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
