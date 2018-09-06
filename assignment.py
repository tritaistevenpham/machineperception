## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
import cv2
import numpy as np

## Main

## Choose: 1 - Read a single file 
##         2 - Read from a directory
fileLoc = './20180826a/SetA/label-7-radioactive-ii.png'
dirLoc = './20180826a/SetA/'

## Option 1
img = cv2.imread( fileLoc, cv2.IMREAD_COLOR)

## Option 2
## img = cv2.imread( dirLoc, cv2.IMREAD_COLOR)

## Color conversion
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

## Detections
laplac = cv2.Laplacian( gray, cv2.CV_64F)
edges = cv2.Canny( gray, 100, 200)


ret, thresh = cv2.threshold(gray, 127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

## Draw Countour
cont = cv2.drawContours(img, contours, -1, (0,255,0), 3)

## Show Image
cv2.imshow( 'Original Image', img)
##cv2.imshow( 'Laplacian', laplac)
cv2.imshow( 'Canny', edges)
cv2.imshow( 'Contour', cont)

## Destroy Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
