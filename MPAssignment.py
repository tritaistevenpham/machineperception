""" COMP3007 - Machine Perception Assignment
        Author: Tri Tai Steven Pham
        Student ID: 17748229
        Year: 2018
"""

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
outputImages = 'Debug/'
"""
    Image directories attempted:
        
        fileLocA = './IMAGES/SetA/*.png'
        fileLocA = './20180826a/SetA/*.png'
        
        fileLocB = './IMAGES/SetB/*.JPG'
        fileLocB = './20180826/SetB/*.JPG'
        
        fileLocC = './IMAGES/SetC/*.JPG'
        fileLocC = './20180826/SetC/*.JPG'
        
        fileLocD = './IMAGES/SetD/*.JPG'
"""
## Current file location for READING images
fileLocB = './Images/SetB/*.JPG'

#Output directories
contourA = 'contourA/'
contourB = 'contourB/'
contourC = 'contourC/'
contourD = 'contourD/'

### START-MAIN
##      To run the program, type in python assignment.py
##

## Option to read 1 file or all files inside a directory, hard-coded directories above
while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1: 
    """ 
        Option 1: Strictly for testing and debugging purposes, not intended for assignment demo.
    """
    
    ## Change filename here (needs to be in working directory
    fileName = 'P1380732.JPG'
    
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
    rows3 = pp.divideImage3( mask)
    pp.readClass( rows3)
    pp.findCharacters( rows3)
    
    cv2.imshow( 'res', mask)
    
elif option == 2:
    """ 
        Option 2: This is the option intended for the assignment demo, it will read all files in
        the specified directory.
    """
    
    ### 
    ## This is the specified file directory, which can be changed at the top of this file.
    ##
    images = glob.glob( fileLocB)
    ##
    ##
    ###
    
    ## Store the images inside an array for processing
    data = []
    filename = []
    
    ## Sort the images in name order, as seen in the actual directory.
    images.sort()
    
    ## For each image in the directory, append to the data[] list for processing
    for files in images:
        img_orig = cv2.imread( files, cv2.IMREAD_COLOR)
        
        # Get fn for output details
        head, fn = os.path.split( files)
        
        img = tf.resizeFunc( img_orig, 900, 900)
        filename.append( fn)
        data.append( img)
        
    for fn, im in zip( filename, data):
        try:
            ## Print filename currently being processed.
            print( fn)
            ## Pre-process original images to get mask of the hazmat labels
            mask = pp.preprocessImage( im)
            mask = cv2.resize( mask, ( 500, 500))
            
            ## Apply the colour balance technique for colour detection
            cb = pp.colourBalance( mask)
            pp.detectHSVColours( cb)
            
            ## Prepare the image subdivisions for character and symbol processing
            rows = pp.divideImage3( mask)
            pp.readClass( rows)
            pp.findCharacters( rows)
            
            ## Write images to a file, used for debugging and testing
            ## cv2.imwrite( ( outputImages + contourB + fn), mask)
            
            ## Symbols have not been implemented yet.
            print( 'symbol: none')
            print( '')
            
        except Exception as e:
            ## Outputs an exception message for debugging
            print(e)
            
cv2.waitKey(0)
cv2.destroyAllWindows

### END-MAIN

