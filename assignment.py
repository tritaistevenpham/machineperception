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
#Image directories
fileLocA = './IMAGES/SetA/*.png'
#fileLocA = './20180826a/SetA/*.png'
fileLocB = './IMAGES/SetB/*.JPG'
#fileLocB = './20180826/SetB/*.JPG'
#fileLocC = './IMAGES/SetC/*.JPG'
fileLocC = './20180826/SetC/*.JPG'
fileLocD = './IMAGES/SetD/*.JPG'

png = '.png'
jpg = '.JPG'

#Output directories
contourA = 'results/contourA/img-'
contourB = 'results/contourB/img-'
contourC = 'results/contourC/img-'
contourD = 'results/contourD/img-'

# Re-size to 900x with AR maximums
maxWidth = 900
maxHeight = 900

### START-RESIZE FUNCTION

def resizeFunc( img):
    imgh, imgw = img.shape[:2]
        
    if( maxHeight <= imgh or maxWidth <= imgw):
        # Re-size for consistency across all images
        r = maxHeight / float( imgh)
        if( maxWidth / float( imgw) < r):
            r = maxWidth / float( imgw)
        
        #dim =  (1000,  int( img_orig.shape[0] * r))
        img = cv2.resize( img_orig, None, fx=r, fy=r, interpolation = cv2.INTER_AREA)
    return img 

### END-RESIZE FUNCTION

### START-MASK DETECTION

def preprocessImage(img):
    ##  First convert the image into grayscale for pre-processing
    img_eq = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    #Create adaptive histogram equalisation to deal with shadows
    clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=( 8, 8))
    cl = clahe.apply(img_eq)
      
    ## Apply a sharpening with a smoothing GaussianBlur on the equalised gray image
    blur = cv2.GaussianBlur( cl, (5,5), 0)
    sharpen = cv2.addWeighted( cl, 0.8, blur, 0.2, 0)
    
    ##  Apply Bilateral Filter on grayscale image & run Canny edge detector
    ##  bilateral params -> 5 for processing, 75, 75 thresholding
    bfilt = cv2.bilateralFilter( sharpen, 5, 75, 75)
    
    ## Canny params -> 255/3, 255 thresholding provides the best result on resized image
    thresh = cv2.Canny( bfilt, 255/3, 255)

    ##  Then dilate the edges to join some undesirable contours
    ##  3x3 kernel of 1's for dilation for just joining important contours
    dilated = cv2.dilate( thresh, np.ones((3, 3), np.uint8), iterations=1)
    
    ##  Find the contours within the computed map; sort by size in desc order
    contours = cv2.findContours( dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ## contours[1] for the version of cv currently running; contours[0] for older versions
    cnt = contours[1]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
    
    ## If we want to draw the bounding rectangle: Set the value of cont to the max in cnt and draw
    # cont = max( cnt, key=cv2.contourArea)
    
    ## Set up mask
    mask = np.zeros( img.shape, np.uint8)
    
    for c in cnt:
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.02 * perimeter, True)
        
        # If the contour detected has 4 vertices; likely to be our sign:
        if ( len( aprx) == 4) and cv2.contourArea( aprx) > float(4000.0):
            sign = aprx
            
        # Draw the signs contour onto the mask and fill
        cv2.drawContours( mask, [sign], -1, (255, 255, 255), -1)

    return mask

### END-MASK DETECTION

### START-MAIN

## Option to read 1 file or all files inside a directory, hard-coded directories above
while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1: ## 4 + shadow P1380524.JPG | 2 P1380513.JPG | 1 P1380502.JPG | B - P1380463.JPG
    fileName = './P1380463.JPG'
    ## Option 1: Read a single file
    img_orig = cv2.imread( fileName, cv2.IMREAD_COLOR)
    
    ## Pre-process original image to get mask of the hazmat labels
    img = resizeFunc( img_orig)
    mask = preprocessImage( img)
    
    ## Minus out the noise to black and show the sign
    res = cv2.bitwise_and( img, mask)
    
    ## Process the mask for expected results
    
elif option == 2:
    ## For now, change this for file location
    images = glob.glob(fileLocB)
    
    ## Store the images inside an array for processing
    data = []
    for files in images:
        img_orig = cv2.imread(files, cv2.IMREAD_COLOR)
        
        img = resizeFunc( img_orig)
        data.append(img)
        
    ## Option 2: Read all image files
    idx = 0
    for im in data:
        try:
            ## Pre-process original images to get mask of the hazmat labels
            mask = preprocessImage(im)
            res = cv2.bitwise_and( im, mask)
            
            ## Process the mask for expected results
            
            ## Change contour output location & image type here
            print(outputImages + contourB + str(idx) + jpg)
            
            cv2.imwrite( (outputImages + contourB + str(idx) + jpg), res)
            idx += 1
        except Exception as e:
            print(e)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows

### END-MAIN

