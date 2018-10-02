## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
from SignDetector import SignDetector
import pytesseract as pt
import cv2
import numpy as np
import glob
import os

## Global declarations
fileLoc = ''
fileName = ''
option = 0
outputImages = 'IMAGES/'
outputDir = 'Output/'

### START-MASK DETECTION

def preprocessImage(img):
    
    img_eq = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    #Create
    clahe = cv2.createCLAHE( clipLimit=3.5, tileGridSize=( 8, 8))
    cl = clahe.apply(img_eq)
    
    ## Color conversion - grayscale for image processing
    #gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    ## Goal: Contour the diamond-shaped hazmat sign to be the main processing space.    
    ## Apply Gaussian Blur and add this weight to the grayscale image
    #cl1 = cv2.medianBlur( cl1, 5)
    blur = cv2.GaussianBlur( cl, (5,5), 0)
    
    #kernel = np.array( [ [0,-1,0], [-1,5,-1], [0,-1,0]])
    #dest = cv2.filter2D( gray_img, -1, kernel)
    
    sharpen = cv2.addWeighted( img_eq, 0.8, blur, 0.2, 0)
    
    
    ##      Apply Bilateral Filter on grayscale image & run Canny edge detector
    bfilt = cv2.bilateralFilter( sharpen, 5, 75, 75)
    thresh = cv2.Canny( bfilt, 255/3, 200)
    dilated = cv2.dilate( thresh, np.ones((3, 3), np.uint8), iterations=1)
    
    ##      Find the contours within the computed edge map; sort by size in desc order
    contours = cv2.findContours( dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
    
    # cont -> for Bounding Rect
    cont = max( cnt, key=cv2.contourArea)
    
    # Initialise signDetect
    #signDetect = SignDetector()
    
    for c in cnt:
        #sign = signDetect.findSign( c)
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.02 * perimeter, True)
        
        # If the contour detected has 4 vertices; likely to be our sign:
        if ( len( aprx) == 4) and cv2.contourArea( aprx) > float(4000.0):
            # Compute the bounding box
            sign = aprx
            
        # Draw the sign contour
        cv2.drawContours( img, [sign], -1, (0, 255, 0), 3)
        
        #cv2.drawContours( img, c, 0, (0, 0, 255), 4)
        
    ## Draw mask
    mask = np.zeros( img.shape, np.uint8)
    #cv2.drawContours( mask, cnt, 0, (255,255,255), 3)
    
    ## Draw contours on img
    #cv2.drawContours( mask, [sign], -1, (0,0,255), 2)
    
    ##      Draw the bounding box for testing 
    #x, y, w, h = cv2.boundingRect( cont)
    #cv2.rectangle( img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    return img

### END-MASK DETECTION

### START-MAIN
    
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
    #Image directories
    fileLocA = './IMAGES/SetA/*.png'
    fileLocB = './IMAGES/SetB/*.JPG'
    fileLocC = './IMAGES/SetC/*.JPG'
    fileLocD = './IMAGES/SetD/*.JPG'
    
    png = '.png'
    jpg = '.JPG'
    
    #Output directories
    contourA = 'results/contourA/img-'
    contourB = 'results/contourB/img-'
    contourC = 'results/contourC/img-'
    contourD = 'results/contourD/img-'
    
    maxWidth = 900
    maxHeight = 900
    
    
    #For now, change this for file location
    images = glob.glob(fileLocC)
    
    data = []
    for files in images:
        img_orig = cv2.imread(files, cv2.IMREAD_COLOR)
        
        imgh, imgw = img_orig.shape[:2]
        
        if( maxHeight <= imgh or maxWidth <= imgw):
            # Re-size for consistency across all images
            r = maxHeight / float( imgh)
            if( maxWidth / float( imgw) < r):
                r = maxWidth / float( imgw)
            
            #dim =  (1000,  int( img_orig.shape[0] * r))
            img = cv2.resize( img_orig, None, fx=r, fy=r, interpolation = cv2.INTER_AREA)
            data.append(img)
        else:
            data.append(img_orig)
    
    ## Option 2
    idx = 0
    for im in data:
        try:
            result = preprocessImage(im)
            #Change contour output location & image type here
            print(outputImages + contourC + str(idx) + jpg)
            
            cv2.imwrite( (outputImages + contourC + str(idx) + jpg), result)
            idx += 1
        except Exception as e:
            print(e)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows

### END-MAIN

