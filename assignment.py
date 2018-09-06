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

blobDetect = cv2.SimpleBlobDetector_create()
keypoints = blobDetect.detect(gray)
im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

## Image Filter
##img_filt = cv2.medianBlur(gray, 5)
##img_th = cv2.adaptiveThreshold( img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
##contours, hierarchy, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ret, thresh = cv2.threshold(gray, 127,255,0)
contours, hierarchy, _ = cv2.findContours(thresh,1,2)

## Draw Countour
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle( gray, (x,y), (x+w,y+h), (0,255,0), 2)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
##img = cv2.drawContours(img, cnt, 0, (0,255,0), 3)

## Show Image
cv2.imshow( 'Original Image', img)
##cv2.imshow( 'Laplacian', laplac)
cv2.imshow( 'Canny', edges)
cv2.imshow( 'Keypoints', im_with_keypoints)

## Destroy Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
