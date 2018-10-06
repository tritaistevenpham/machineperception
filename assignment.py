## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
import pytesseract as pt
import cv2
import numpy as np
import glob
import os

## Import user-defined functions
import PreprocessFunctions as pp
import TransformFunctions as tf

## Global declarations
option = 0
outputImages = 'IMAGES/'
outputDir = 'Output/'
## Image directories
#fileLocA = './IMAGES/SetA/*.png'
fileLocA = './20180826a/SetA/*.png'
#fileLocB = './IMAGES/SetB/*.JPG'
fileLocB = './20180826/SetB/*.JPG'
#fileLocC = './IMAGES/SetC/*.JPG'
fileLocC = './20180826/SetC/*.JPG'
fileLocD = './IMAGES/SetD/*.JPG'

#Output directories
contourA = 'results/contourA/'
contourB = 'results/contourB/'
contourC = 'results/contourC/'
contourD = 'results/contourD/'

### START-MAIN

## Option to read 1 file or all files inside a directory, hard-coded directories above
while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1: ## 4 + shadow P1380524.JPG | 2 P1380513.JPG | 1 P1380502.JPG | B - P1380463.JPG
    #fileName = 'label-7-radioactive-ii.png' #YELLOWWHITE
    #fileName = 'label-8-corrosive.png' #BLACKWHITE
    #fileName = 'label-4-dangerous-when-wet.png'#BLUE
    #fileName = 'label-2-non-flammable-gas.png' #GREEN
    #fileName = 'P1380463.JPG' #green B
    fileName = 'P1380475.JPG' #orange B
    ## Option 1: Read a single file
    img_orig = cv2.imread( fileName, cv2.IMREAD_COLOR)
    
    ## Pre-process original image to get mask of the hazmat labels
    img = tf.resizeFunc( img_orig, 900, 900)
    mask = pp.preprocessImage( img)
    
    ## Re-size the transform to work in a smaller but consistent space (same as Set A sizes)
    mask = cv2.resize( mask, ( 500, 500))
    
    ## Prepare colour balance and detect the colours
    cb = pp.colourBalance( mask)
    pp.detectHSVColours( cb)
    
    ## Prepare the image subdivisions for character and symbol processing
    rows = pp.divideImage( mask)
    pp.readClass( rows)
    
    #pp.readSign( mask)
    
    cv2.imshow( 'res', mask)
    ## Perform perspective transform on the mask for expected working space
    #points = getPoints( res)
    
    
elif option == 2:
    ## For now, change this for file location
    images = glob.glob( fileLocB)
    
    ## Store the images inside an array for processing
    data = []
    filename = []
    images.sort()
    for files in images:
        img_orig = cv2.imread( files, cv2.IMREAD_COLOR)
        
        # Get fn for output details
        head, fn = os.path.split( files)
        
        img = tf.resizeFunc( img_orig, 900, 900)
        filename.append( fn)
        data.append( img)
        
    ## Option 2: Read all image files
    idx = 0
    
    for fn, im in zip( filename, data):
        try:
            print( fn)
            ## Pre-process original images to get mask of the hazmat labels
            mask = pp.preprocessImage( im)
            mask = cv2.resize( mask, ( 500, 500))
            ## Segment the image into 5 rows; 1st row, 5th row detects colour
            ##  1st, 2nd, 3rd row detects symbol
            ##  2nd, 3rd, 4th row detects words
            ##  1st, 2nd, 3rd, 5th row detects class(es)
            #blur = pp.findColours( mask)
            cb = pp.colourBalance( mask)
            pp.detectHSVColours( cb)
            #pp.readSign( mask)
            ## Prepare the image subdivisions for character and symbol processing
            rows = pp.divideImage( mask)
            pp.readClass( rows)
            ## Change contour output location & image type here
            #print( outputImages + contourB + str(idx) + jpg)
            #print( outputImages + contourA + fn)
            
            cv2.imwrite( ( outputImages + contourB + fn), mask)
            print( '')
            #cv2.imwrite( ( outputImages + contourB + str(idx) + jpg), res)
            #idx += 1
        except Exception as e:
            print(e)
            
cv2.waitKey(0)
cv2.destroyAllWindows

### END-MAIN

