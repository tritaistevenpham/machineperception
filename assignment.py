## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
import cv2
import numpy as np
import glob
import os

## Main
fileLoc = ''
fileName = ''
option = 0
outputDir = 'Output/'
## Choose: 1 - Read a single file from SetA
##         2 - Read all images from a directory

def preprocessImage(img):
    ## Color conversion - grayscale for image processing
    

    ### OBJECT DETECTION
    ## Goal: Contour the diamond-shaped hazmat sign to be the main processing space.

    ##      Apply Gaussian blur with a 5 x 5 kernel to the image (Reduce high frequency noise)
    
    
    ##      Apply Bilateral Filter on grayscale image
    gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    bfilt = cv2.bilateralFilter( gray_img, 5, 85, 75)
    
    thresh = cv2.Canny( bfilt, 255/3, 255)
    
    #edge = cv2.Laplacian( gblur, cv2.CV_8UC1)

    ##      Find the contours within the computed edge map; sort by size in desc order
    contours = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
    displayCnt = None

    ##      Loop over the contours
    aprx = None
    sign_arr = []
    rect_coords = []
    running_count = 0
            
    cont = max( cnt, key=cv2.contourArea)
    #cv2.drawContours( img, cont, -1, (255,255,0), 8)
    cv2.drawContours( img, cnt, 0, (0,0,255), 5)
    
    x, y, w, h = cv2.boundingRect( cont)
    cv2.rectangle( img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    ### END OBJECT DETECTION

    return img

while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1: ## 4 + shadow P1380524.JPG | 2 P1380513.JPG | 1 P1380502.JPG 
    fileName = './P1380513.JPG'
    ## Option 1 + resize
    img_orig = cv2.imread( fileName, cv2.IMREAD_COLOR)
    
    r = 600.0 / img_orig.shape[1]
    dim =  (600,  int( img_orig.shape[0] * r))

    img = cv2.resize( img_orig, dim, interpolation = cv2.INTER_AREA)
    preprocessImage(img)
    
elif option == 2:
    fileLocA = './20180826a/SetA/*.png'
    fileLocB = './20180826/SetB/*.JPG'
    fileLocC = './20180826/SetC/*.JPG'
    
    png = '.png'
    jpg = '.JPG'
    
    contourA = 'contourA/img-'
    contourB = 'contourB/img-'
    contourC = 'contourC/img-'
    
    #validImgPath = os.path.join(fileLocB, '*.jpg')
    images = glob.glob(fileLocC)
    
    data = []
    for files in images:
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        data.append(img)
    
    ## Option 2
    idx = 0
    for im in data:
        try:
            result = preprocessImage(im)
            print(outputDir + contourC + str(idx) + jpg)
            cv2.imwrite( (outputDir + contourC + str(idx) + jpg), result)
            idx += 1
        except Exception as e:
            print(e)



