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
contourA = 'results/contourA/'
contourB = 'results/contourB/'
contourC = 'results/contourC/'
contourD = 'results/contourD/'

# Re-size to 900x with AR maximums
maxWidth = 900
maxHeight = 900

### START-GET POINTS FUNCTION

#def getPoints( img):
    

### END-GET POINTS FUNCTION

### START-ORDER POINTS FUNCTION

def orderPoints( points):
    ## Initialise a set of 4 points in TL->TR->BR->BL ordering
    rect = np.zeros( ( 4, 2), dtype = "float32")
    
    ## With this system, top left with have smallest sum of XY
    ## and bottom right will have the largest sum of XY
    
    tot = points.sum( axis = 1)
    rect[ 0] = points[ np.argmin( tot)]
    rect[ 2] = points[ np.argmax( tot)]
    
    ## Top right will have the smallest difference between points
    ## Bottom left will have the largest difference between points
    diff = np.diff( points, axis = 1)
    rect[ 1] = points[ np.argmin( diff)]
    rect[ 3] = points[ np.argmax( diff)]
    return rect;
    
### END-ORDER POINTS FUNCTION

### START-FOUR POINT TRANSFORM FUNCTION

def fourPtTrans( img, points):
    ## Unpack the points and keep them consistent:
    #rect = orderPoints( points)
    rect = points #imported as [0] left, [1] right, [2] top, [3] bottom
    ( left, right, top, bot) = rect
    
    ## Unpack the width and height of the img
    imgHeight, imgWidth = img.shape[ :2]
    
    ## Specify destination points (top-left, top-right, bottom-right, bottom-left)
    ## Into diamond (TL = top, TR = right, BR = bottom, BL = left)
    dest1 = np.array( [ 
        [ (imgWidth / 2.0), 0], 
        [ imgWidth, (imgHeight / 2.0)], 
        [ imgWidth, imgHeight], 
        [0, imgHeight]], dtype = "float32")
    
    ## Into diamond, ordering consistent as above
    dest = np.array( [
        [ 0, (imgHeight / 2.0)], #left
        [ imgWidth, (imgHeight / 2.0)], #right
        [ (imgWidth / 2.0), 0], #top
        [ (imgWidth) / 2.0, imgHeight]], dtype = "float32") #bottom
    
    ## Compute the transformation matrix and apply
    T = cv2.getPerspectiveTransform( rect, dest)
    warp = cv2.warpPerspective( img, T, ( imgWidth, imgHeight))
    
    return warp

### END-FOUR POINT TRANSFORM FUNCTION
    
### START-RESIZE FUNCTION

def resizeFunc( img):
    imgh, imgw = img.shape[ :2]
        
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

def preprocessImage( img):
    ##  First convert the image into grayscale for pre-processing
    img_eq = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    #Create adaptive histogram equalisation to deal with shadows
    clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=( 8, 8))
    cl = clahe.apply( img_eq)
      
    ## Apply a sharpening with a smoothing GaussianBlur on the equalised gray image
    blur = cv2.GaussianBlur( cl, ( 5, 5), 0)
    sharpen = cv2.addWeighted( cl, 0.8, blur, 0.2, 0)
    
    ##  Apply Bilateral Filter on grayscale image & run Canny edge detector
    ##  bilateral params -> 5 for processing, 75, 75 thresholding
    bfilt = cv2.bilateralFilter( sharpen, 5, 75, 75)
    
    ## Canny params -> 255/3, 255 thresholding provides the best result on resized image
    thresh = cv2.Canny( bfilt, 255/3, 255)

    ##  Then dilate the edges to join some undesirable contours
    ##  3x3 kernel of 1's for dilation for just joining important contours
    dilated = cv2.dilate( thresh, np.ones( ( 3, 3), np.uint8), iterations=1)
    
    ##  Find the contours within the computed map; sort by size in desc order
    contours = cv2.findContours( dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ## contours[1] for the version of cv currently running; contours[0] for older versions
    cnt = contours[ 1]
    cnt = sorted( cnt, key=cv2.contourArea, reverse=True)
    
    ## If we want to draw the bounding rectangle: Set the value of cont to the max in cnt and draw
    # cont = max( cnt, key=cv2.contourArea)
    
    ## Set up mask
    mask = np.zeros( img.shape, np.uint8)
    
    ## Set up points to transform
    points = np.zeros( ( 4, 2), dtype = "float32")
    
    for c in cnt:
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.02 * perimeter, True)
        
        # If the contour detected has 4 vertices; likely to be our sign:
        if ( len( aprx) == 4) and cv2.contourArea( aprx) > float( 4000.0):
            sign = aprx
            ## Find the extreme corners of the detected sign and store them for P.T.
            # Extreme left point
            points[0] = tuple( c[ c[ :,:,0].argmin()][0])
            # Extreme right point
            points[1] = tuple( c[ c[ :,:,0].argmax()][0])
            # Extreme top point
            points[2] = tuple( c[ c[ :,:,1].argmin()][0])
            # Extreme bot point
            points[3] = tuple( c[ c[ :,:,1].argmax()][0])
            
        # Draw the signs contour onto the mask and fill
        cv2.drawContours( mask, [sign], -1, ( 255, 255, 255), -1)
        ## Minus out the noise to black and show the sign
        res = cv2.bitwise_and( img, mask)
        
        warp = fourPtTrans( res, points)

    return warp

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
    
    cv2.imshow( 'res', mask)
    ## Perform perspective transform on the mask for expected working space
    #points = getPoints( res)
    
    
elif option == 2:
    ## For now, change this for file location
    images = glob.glob( fileLocB)
    
    ## Store the images inside an array for processing
    data = []
    filename = []
    for files in images:
        img_orig = cv2.imread( files, cv2.IMREAD_COLOR)
        
        # Get fn for output details
        head, fn = os.path.split( files)
        
        img = resizeFunc( img_orig)
        filename.append( fn)
        data.append( img)
        
    ## Option 2: Read all image files
    idx = 0
    for im, fn in zip( data, filename):
        try:
            ## Pre-process original images to get mask of the hazmat labels
            mask = preprocessImage( im)
            
            ## Process the mask for expected results
            
            ## Change contour output location & image type here
            #print( outputImages + contourB + str(idx) + jpg)
            print( outputImages + contourB + fn)
            
            cv2.imwrite( ( outputImages + contourB + fn), mask)
            #cv2.imwrite( ( outputImages + contourB + str(idx) + jpg), res)
            #idx += 1
        except Exception as e:
            print(e)
            
cv2.waitKey(0)
cv2.destroyAllWindows

### END-MAIN

