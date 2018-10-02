###
# Adapted from the Shape Detector implementation:
#   pyimagesearch.com/2016/02/08/opencv-shape-detection
###

import cv2

class SignDetector:
    def __init__(self):
        pass
    
    def findSign( self, c):
        signFound = False
        # Approximate the CONTOUR
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.04 * perimeter, True)
        
        # If the contour detected has 4 vertices; likely to be our sign:
        if( len( aprx) == 4):
            # Compute the bounding box
            (x, y, w, h) = cv2.boundingRect( aprx)
            # Compute the aspect ratio
            asr = w / float(h)
            signFound = True
            
        return signFound
