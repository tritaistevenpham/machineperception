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
    fileLoc = './20180826a/SetA/label-7-radioactive-ii.png'
elif option == 2:
    fileLoc = './20180826a/SetA/'

## Option 1
img = cv2.imread( fileLoc, cv2.IMREAD_COLOR)

## Option 2
## img = cv2.imread( dirLoc, cv2.IMREAD_COLOR)

## Color conversion - grayscale for image processing
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

## Show Image
cv2.imshow( 'Original Image', img)

## Destroy Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
