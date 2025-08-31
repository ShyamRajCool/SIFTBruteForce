import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt


siftAlg=cv.SIFT_create()

shuttleImage1=cv.imread('spaceShuttle2.jpg')
shuttleImage2=cv.imread('shuttle.jpg')

keypoint1,descriptorVector1=siftAlg.detectAndCompute(shuttleImage1,None)
keypoint2,descriptorVector2=siftAlg.detectAndCompute(shuttleImage2,None)

bruteForceMethod=cv.BFMatcher()
const_K=2 #K, to my knowledge, are the neighbors who have the shortest distance to the descriptor(s)
matches=bruteForceMethod.knnMatch(descriptorVector1,descriptorVector2,const_K)

validBestMatches = []
print(matches[0])


for bestMatch, secondBestMatch in matches:
    if bestMatch.distance < 0.7 * secondBestMatch.distance:
        validBestMatches.append([bestMatch])

print(len(validBestMatches))
lineDrawImg = cv.drawMatchesKnn(shuttleImage1, keypoint1, shuttleImage2,keypoint2, validBestMatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure() #Think of it as the background
plt.imshow(lineDrawImg) #Essentially creates the image
plt.show() #Creates a nice window that you can zoom-in and zoom out of of
