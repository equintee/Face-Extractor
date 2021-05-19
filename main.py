import cv2 as cv
import os
#Emirhan UREY @equintee

#Getting image as source.
src = cv.imread('src.jpg')
#Converting it to gray scale.
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

#Loading up face detection algorithm.
haar_cascade = cv.CascadeClassifier('haar_face.xml')

#Running the algorithm.
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

#Giving console output for how many faces that program found.
print(f'Number of faces found = {len(faces_rect)}')

#Empty list for faces.
face_list = []

#A loop for that crops faces from image and appends it on the list.
for (x,y,w,h) in faces_rect:
    face_list.append(src[y:y+h, x:x+w])

#A loop for to save extracted faces to working directory.
i = 1
for faces in face_list:
    cv.imwrite(f"{i}.jpg", faces)
    i += 1

