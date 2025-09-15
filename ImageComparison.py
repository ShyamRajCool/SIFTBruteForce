import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import os


file1 = os.path.join(os.path.dirname(__file__),'shuttle_01.jpg')
file2 = os.path.join(os.path.dirname(__file__),'shuttle_03.jpg')

siftAlg = cv.SIFT_create()

shuttleImage1 = cv.imread(file1,-1)
shuttleImage2 = cv.imread(file2,-1)



keypoint1,descriptorVector1=siftAlg.detectAndCompute(shuttleImage1,None)
keypoint2,descriptorVector2=siftAlg.detectAndCompute(shuttleImage2,None)

bruteForceMethod=cv.BFMatcher()
const_K=2 #K, to my knowledge, are the neighbors who have the shortest distance to the descriptor(s)
matches=bruteForceMethod.knnMatch(descriptorVector1,descriptorVector2,const_K)

validBestMatches = []
print(matches[0])


for bestMatch, secondBestMatch in matches:
    if bestMatch.distance < .7 * secondBestMatch.distance:
        validBestMatches.append([bestMatch])

print(len(validBestMatches))
lineDrawImg = cv.drawMatchesKnn(shuttleImage1, keypoint1, shuttleImage2,keypoint2, validBestMatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure() #Think of it as the background
plt.imshow(lineDrawImg) #Essentially creates the image
plt.show() #Creates a nice window that you can zoom-in and zoom out of of
