import cv2
import numpy as np

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

def resizeFunc( img_orig, max_h, max_w):
    img = img_orig.copy()
    imgh, imgw = img.shape[ :2]
        
    if( max_h <= imgh or max_w <= imgw):
        # Re-size for consistency across all images
        r = max_h / float( imgh)
        if( max_w / float( imgw) < r):
            r = max_w / float( imgw)
        
        img = cv2.resize( img_orig, None, fx=r, fy=r, interpolation = cv2.INTER_AREA)
    return img 

### END-RESIZE FUNCTION
