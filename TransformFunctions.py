""" COMP3007 - Machine Perception Assignment
        Author: Tri Tai Steven Pham
        Student ID: 17748229
        Year: 2018
"""

## Import libraries
import cv2
import numpy as np

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

def resizeFunc( img_orig, max_h, max_w):
    img = img_orig.copy()
    imgh, imgw = img.shape[ :2]
        
    if( max_h <= imgh or max_w <= imgw):
        # Re-size for consistency across all images
        scale = max_h / float( imgh)
        if( max_w / float( imgw) < scale):
            scale = max_w / float( imgw)
        
        img = cv2.resize( img_orig, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    return img

### END-RESIZE FUNCTION

### START-SORT CONTOURS FUNCTION

def getContPrec( contour, cols):
    origin = cv2.boundingRect( contour)
    return ( ( origin[1] // 5) * 5 ) * cols + origin[ 0]

def sortContours( cnts):
    ## Initialise reverse flag and index
    reverse = True
    idx = 1
    
    ## Construct the bounding boxes:
    bounding = [ cv2.boundingRect( c) for c in cnts]
    ( cnts, bounding) = zip( *sorted( zip( cnts, bounding), key=lambda b:b[1][idx], reverse=reverse))
    
    return (cnts, bounding)

### END-SORT CONTOURS FUNCTION
