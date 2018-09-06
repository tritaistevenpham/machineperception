## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
import cv2
import numpy as np

## Main
fileLoc = ''
option = 0
## Choose: 1 - Read a single file from SetA
##         2 - Read all images from a directory

while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file from SetA\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1:
    fileLoc = './P1380502.JPG'
elif option == 2:
    fileLoc = './20180826a/SetA/label-7-radioactive-ii.png'

## Option 1 + resize
img_orig = cv2.imread( fileLoc, cv2.IMREAD_COLOR)

r = 500.0 / img_orig.shape[1]
dim =  (500,  int( img_orig.shape[0] * r))

img = cv2.resize( img_orig, dim, interpolation = cv2.INTER_AREA)

## Option 2
## img = cv2.imread( dirLoc, cv2.IMREAD_COLOR)

## Color conversion - grayscale for image processing
gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

### SHAPE DETECTION
## Goal: Contour the diamond-shaped hazmat sign to be the main processing space.

##      Apply Gaussian blur with a 5 x 5 kernel to the image (Reduce high frequency noise)
gblur = cv2.GaussianBlur( gray_img, (5, 5), 0)

##      Compute the edge map
edge = cv2.Canny( gblur, 50, 200, 255)

##      Find the contours within the computed edge map; sort by size in desc order
contours = cv2.findContours( edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
displayCnt = None

##      Loop over the contours
aprx = None
for c in cnt:
    ## Approximate the contour
    al =  cv2.arcLength( c, True)
    aprx = cv2.approxPolyDP( c, 0.02 * al, True)
    
    # Use bounding rectangle on the approximated points:
    rect = cv2.boundingRect( aprx)
    x, y, w, h = rect
    cv2.rectangle( img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    print('Approx coordinates: ', aprx)
    ## If the contour has 4 vertices, we have found "a" diamond shape.
    if len( aprx) == 4:
        displayCnt = aprx
        print('found 4 vertices')
        break

##      Draw detected "corners"
cv2.drawContours( img, aprx, -1, (0,255,0), 5)

##      Finding the four vertices means we can extract the contents
##      Extract the sign, and apply a PERSPECTIVE transform
pts1 = np.float32( [[aprx[0]], [aprx[1]], [aprx[2]], [aprx[3]]])
pts2 = np.float32( [[250,0], [0,250], [250,500], [500, 250]])

### SHAPE DETECTION FINISH

## Show Images
cv2.imshow( 'Original Image', img)
cv2.imshow( 'Edge Map', edge)

## Destroy Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
