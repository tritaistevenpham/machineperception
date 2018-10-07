""" COMP3007 - Machine Perception Assignment
        Author: Tri Tai Steven Pham
        Student ID: 17748229
        Year: 2018
"""

## Import libraries
import cv2
import numpy as np

## Import user-defined functions
import TransformFunctions as tf
import pytesseract as pt
import math as m

### START-MASK DETECTION
##      This function is used to eliminate the background from the hazmat label
##      It uses a series of filters and smoothing techniques to filter out the background
##      It was almost prepared to tackle Set C however it now only perfectly detects 
##      items from Set B (and Set A)
##

def preprocessImage( img):
    ##  First convert the image into grayscale for pre-processing
    warp = img.copy()
    img_eq = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    ## Create adaptive histogram equalisation to deal with shadows
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
    
    """ I have decided to analyse the contours and make a judgement on approximated
        points. I have chosen to look for 4 equal points approximated from each other and 
        declared them as the points of a square - or the sign in question.
    """
    for c in cnt:
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.02 * perimeter, True)
        
        # If the contour detected has 4 vertices; likely to be our sign:
        if ( len( aprx) == 4) and cv2.contourArea( aprx) > float( 4000.0):
            sign = aprx
            ## Find the extreme-corners of the detected sign and store them for P.T.
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
            
            ## Perform the transformation method
            warp = tf.fourPtTrans( res, points)
        
    return warp

### END-MASK DETECTION

### START-PREPROCESSING FOR COLOUR FUNCTION
##      colourBalance(): This function was adapted from the forum post below
##          It's aim is to balance the colours across the image so the image
##          isn't too dark or too bright for the program to process 
##

def colourBalance( img):
    ## Function for colour balancing from:
    ## https://stackoverflow.com/a/46391574 - norok2 (Sep 24 '17)
    
    ## Use the LAB colourspace to balance the image luminosity
    ret_img = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    
    ## Use NumPy's average function to find the average for A and B of the image
    avg1 = np.average( ret_img[ :, :, 1])
    avg2 = np.average( ret_img[ :, :, 2])
    
    ## Apply an averaging calculation on the output image:
    ## Image pixel - ( ( average - 128) * image pixel luminosity / 255) * 1.1 scalar )
    ret_img[ :, :, 1] = ret_img[ :, :, 1] - ( ( avg1 - 128) * ( ret_img[ :, :, 0] / 255.0) * 1.1)
    ret_img[ :, :, 2] = ret_img[ :, :, 2] - ( ( avg2 - 128) * ( ret_img[ :, :, 0] / 255.0) * 1.1)
    ret_img = cv2.cvtColor( ret_img, cv2.COLOR_LAB2BGR)
    
    return ret_img

##
##      detectHSVColours(): This function is responsible for detecting the colours
##          of the hazmat label signs. Just the basic pre-processing techniques were used here.
##

def detectHSVColours( img):
    
    ## Give the image a blur to filter out small noise
    blurred = cv2.GaussianBlur( img, (5,5), 3)
    
    ## Help the smoothing with a morphology erosion and dilation
    opened = cv2.morphologyEx( img, cv2.MORPH_OPEN, np.ones( ( 5, 5), np.uint8))
    
    ## Use the HSV colourspace to classify colours
    hsv_img = cv2.cvtColor( opened, cv2.COLOR_BGR2HSV)
    
    ## Select likely top and bottom of the image manually: img[ y, x]
    colour1 = hsv_img[ 190, 395]
    colour2 = hsv_img[ 370, 190]
    
    ## Initialise colour output
    topShow = 'top:'
    botShow = 'bot:'
    actualTop = ''
    actualBot = ''
    
    ## Colour HSV range thresholding to find the correct values to classify the data set based
    ## on Simple Background Images - these were handpicked HSV values to cater towards Set B
    
    ## Check for White
    if (colour1[0] >= 0 and colour1[0] <= 24) or (colour1[0] >= 70 and colour1[0] <= 120):
        if (colour1[1] >= 0 and colour1[1] <= 85):
            if (colour1[2] >= 120 and colour1[2] <= 130) or colour1[2] >= 141:
                actualTop = ' white'
        
    if (colour2[0] >= 0 and colour2[0] <= 24) or (colour2[0] >= 70 and colour2[0] <= 120):
        if (colour2[1] >= 0 and colour2[1] <= 85):
            if (colour2[2] >= 120 and colour2[2] <= 130) or colour2[2] >= 141:
                actualBot = ' white'
                
    ## Check for Green
    if colour1[0] >= 20 and colour1[0] <= 90:
        if colour1[1] >= 70 and colour1[1] <= 130:
            actualTop = ' green'
        
    if colour2[0] >= 20 and colour2[0] <= 90:
        if colour2[1] >= 70 and colour2[1] <= 130:
            actualBot = ' green'
            
    ## Check for Orange
    if (colour1[2] >= 144 and colour1[2] <= 255):
        if (colour1[0] >= 4 and colour1[0] <= 15) or (colour1[0] >= 176 and colour1[0] <= 179):
            if colour1[1] >= 70:
                actualTop = ' orange'
                
    if (colour2[2] >= 144 and colour2[2] <= 255):    
        if (colour2[0] >= 4 and colour2[0] <= 15) or (colour2[0] >= 176 and colour2[0] <= 179):
            if colour2[1] >= 70:
                actualBot = ' orange'
    
    ## Check for Red
    if (colour1[0] >= 160 and colour1[0] <= 179) or (colour1[0] >= 0 and colour1[0] <= 5):
        if colour1[1] >= 110 and colour1[1] <= 178:
            if colour1[2] <= 197:
                actualTop = ' red'
        
    if (colour2[0] >= 160 and colour2[0] <= 179) or (colour2[0] >= 0 and colour2[0] <= 5):
        if colour2[1] >= 110 and colour2[1] <= 178:
            if colour2[2] <= 197:
                actualBot = ' red'
    
    ## Check for Blue
    if colour1[0] >= 105 and colour1[0] <= 125:
        if (colour1[1] >= 0 and colour1[1] <= 50) or (colour1[1] >= 125 and colour1[1] <= 200):
            if colour1[2] >= 108 and colour1[2] <= 132:
                actualTop = ' blue'
    
    if colour2[0] >= 105 and colour2[0] <= 125:
        if (colour2[1] >= 0 and colour2[1] <= 50) or (colour2[1] >= 125 and colour2[1] <= 200):
            if colour2[2] >= 108 and colour2[2] <= 132:
                actualBot = ' blue'
        
    ## Check for Black
    if colour1[0] >= 10 and colour1[0] <= 55:
        if colour1[2] >= 0 and colour1[2] <= 110:
            actualTop = ' black'
    
    if colour2[0] >= 10 and colour2[0] <= 55:
        if colour2[2] >= 0 and colour2[2] <= 110:
            actualBot = ' black'
            
    ## Check for Yellow
    if colour1[0] >= 14 and colour1[0] <= 30:
        if colour1[1] >= 115 and colour1[1] <= 213:
            actualTop = ' yellow'
            
    if colour2[0] >= 14 and colour2[0] <= 30:
        if colour2[1] >= 115 and colour2[1] <= 213:
            actualBot = ' yellow'
    
    topShow += ''.join( actualTop)
    botShow += ''.join( actualBot)
    print( topShow)
    print( botShow)

### END-PREPROCESSING FOR COLOUR FUNCTION

### START-DIVIDE IMAGE FUNCTION
##      These two functions served the purpose of segmenting the image, to reduce
##      the processing power and time by processing on a smaller scale, where 
##      things we know are not useful are excluded.
##

def divideImage3( img_orig):
    ## Set the img width and height
    crop_0 = img_orig.copy()
    imgh, imgw = crop_0.shape[ :2]
    
    ## Calculate the segments into thirds:
    secHeight = int( m.floor( imgh // 3))
    
    ## Create sections to begin cropping the image (for 3 row sections)
    crop_1 = crop_0[ secHeight:, :]
    crop_2 = crop_1[ secHeight:, :]
    
    ## Create the sections and store them as new images for processing
    sec1 = crop_0[ :secHeight, :]
    sec2 = crop_1[ :secHeight, :]
    sec3 = crop_2[ :secHeight, :]
    
    ## Store rows for grouped access
    rows = [sec1, sec2, sec3]
    return rows

def divideImage5( img_orig):
    ## Set the img width and height
    crop_0 = img_orig.copy()
    imgh, imgw = crop_0.shape[ :2]
    
    ## Calculate the segments into fifths:
    secHeight = int( m.floor( imgh // 5))
    
    ## Create sections to begin cropping the image (for 5 row sections)
    crop_1 = crop_0[ secHeight:, :]
    crop_2 = crop_1[ secHeight:, :]
    crop_3 = crop_2[ secHeight:, :]
    crop_4 = crop_3[ secHeight:, :]
    
    ## Create the sections and store them as new images for processing
    sec1 = crop_0[ :secHeight, :]
    sec2 = crop_1[ :secHeight, :]
    sec3 = crop_2[ :secHeight, :]
    sec4 = crop_3[ :secHeight, :]
    sec5 = crop_4[ :secHeight, :]
    
    ## Store rows for grouped access
    rows = [sec1, sec2, sec3, sec4, sec5]
    return rows
    
### END-DIVIDE IMAGE FUNCTION

### START-OCR TEMPLATE CONTOUR
##      This function is responsible for sorting the input template from
##      left to right, and keeping the order. It assumes that the image
##      passed in will be the template image and has been segmented and 
##      concatenated for the purposes of this program; that each contour
##      will be expected to be along one horizontal line in the image. I have
##      manually generated these template images to be so.
##

def findContourOCR( img):
    ## Find Contours on the OCR Template for class numbers:
    templ_ref = cv2.cvtColor( img.copy(), cv2.COLOR_BGR2GRAY)
    
    ## Threshold the template image
    templ_ref = cv2.threshold( templ_ref, 10, 255, cv2.THRESH_BINARY_INV)[ 1]
    ocr_cont = cv2.findContours( templ_ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ocr_cont = ocr_cont[1]
    
    ## Sort the OCR Template contours from left to right using getContPrec
    ocr_cont.sort(key=lambda x:tf.getContPrec(x, templ_ref.shape[ 1]))
    
    ## Initialise a digits set which match digit to roi - {} dictionary assignment
    char = {}
    
    ## Loop over the OCR Template
    for( i, c) in enumerate( ocr_cont):
        ## Compute each digit's bounding box, extract it, and resize appropriately
        ( x, y, w, h) = cv2.boundingRect( c)
        ## Pad the template 
        roi = templ_ref[ y-5:y+h+5, x-5:x+w+5]
        roi = cv2.resize( roi, ( 57, 88))
        
        char[ i] = roi
        
    return char

### END-OCR TEMPLATE CONTOUR

### START-OCR ACTUAL IMAGE CONTOUR
##      This function performs a contour search on a given image
##      It assumes pre-processing has been done and aims to find the 
##      bounding boxes of characters
##

def findContourEachDigit( img):
    ## Find contours in the sectioned image
    img_cnts = cv2.findContours( img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cnts = img_cnts[1]
    
    locs = []
    
    ## Find the bounding rectangles for the img
    for( i, c) in enumerate( img_cnts):
        ( x, y, w, h) = cv2.boundingRect( c)
        
        ## Check if contour pixel height is sensible for a number
        if( w > 10 and w < 80) and ( h > 30 and h < 100):
            ## Also check the area is sensible (considering the size as well)
            if cv2.contourArea( c) > 200:
                ## Draw the rectangle to the image and append the coordinates to locs[]
                cv2.rectangle( img, ( x, y), ( x + w, y + h), ( 0, 255, 0), 1)
                locs.append( ( x, y, w, h))
                
    ## Sort the detected contours from left to right
    locs = sorted( locs, key=lambda x:x[0])
    
    return locs

##
### END-OCR ACTUAL IMAGE CONTOUR

### START-OCR ACTUAL IMAGE CONTOUR
##      This function performs a contour search on a given image
##      It assumes pre-processing has been done and aims to find the 
##      bounding boxes of characters. It is the same function as above
##      But with different parameters. Instead of parsing them in it was
##      easier to test for me to hard code them in the function. Only
##      testing on Set B so the thresholding is catered to that set.
##

def findContourEachChar( img):
    ## Perform a small erosion to get rid of any noise left 
    img = cv2.erode( img, np.ones( ( 3, 3), np.uint8), iterations=1)
    
    ## Find the contours
    img_cnts = cv2.findContours( img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cnts = img_cnts[1]
    
    locs = []
    
    for( i, c) in enumerate( img_cnts):
        ( x, y, w, h) = cv2.boundingRect( c)
        
        ## Check if contour pixel height is sensible for a character
        if( w > 5 and w < 80) and ( h > 20 and h < 100):
            ## If the area is also suitable:
            if cv2.contourArea( c) > 90:
                ## Draw the rectangle and append the region to locs[]
                cv2.rectangle( img, ( x, y), ( x + w, y + h), ( 255, 255, 255), 1)
                locs.append( ( x, y, w, h))
                
    ## Sort the detected contours from left to right
    locs = sorted( locs, key=lambda x:x[0])
    #cv2.imshow( 'contour full roi', img)
    return locs

##
### END-OCR ACTUAL IMAGE CONTOUR

### START-CLASS CLASSIFICATION

def readClass( rows):
    
    ## Find the digits of the template
    templateFile = cv2.imread( './Template/0to9ArialBlack.PNG', cv2.IMREAD_COLOR)
    digits = findContourOCR( templateFile)
    
    ## Use the last row of the segmented image
    img = rows[2]
    img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur( img, 9)
    
    ## Apply adaptive thresholding to get the class number:
    img = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    ## If needed; dilate erosion filters
    # img = cv2.dilate( img, np.ones( ( 5, 5), np.uint8), iterations=1)
    # img = cv2.erode( img, np.ones( ( 5, 5), np.uint8), iterations=1)
    
    ## If image is more white than black, invert, else leave as is
    num_white = np.sum( img == 255)
    if num_white > 2500: 
        img = cv2.bitwise_not(img)
    
    ## Find the contour of each character in the image
    locs = findContourEachDigit( img)
    
    ## @output = will show the extracted number
    ## @black = is the colour of the padding
    ## @pad = value for border padding
    output = []
    black = [0, 0, 0]
    pad = 20
    ## Loop over the existing potential numbers:
    for( i, ( lX, lY, lW, lH)) in enumerate( locs):
        
        ## Extract the digit from binary section image, zoom out a tiny bit
        roi = img[ lY - 5: lY + lH + 5, lX - 5: lX + lW + 5]
        
        ## Initialise a list of template matching scores
        scores = []
        
        if roi.shape[0] and roi.shape[1] > 0:
            ## Ensure ROI is the same size as the template thresholding for scoring
            roi = cv2.resize( roi, ( 57, 88))
            
            ## If the white area is too great, invert the image to score against template
            roi_white = np.sum( roi == 255)
            
            ## Pad the border to make template matching a bit easier
            roi = cv2.copyMakeBorder( roi.copy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=black)
            if roi_white > 2000:
                roi = cv2.bitwise_not(roi)
            
            ## Loop over reference digit and digit ROI:
            for( digit, digitROI) in digits.items():
                ## Apply template matching and score on max value of match
                result = cv2.matchTemplate( roi, digitROI, cv2.TM_CCOEFF)
                ( _, score, _, _) = cv2.minMaxLoc( result)
                scores.append( score)
                
            ## Scores should match with template contours -> already sorted     
            output.append( str( np.argmax( scores)))
            
    ## Update the digits output list
    if len( output) > 0 and len( output) < 3:
        ## If the output has 2 digits found, assume a decimal
        print( 'class: {}'.format( '.'.join( output)))
    else:
        ## Any less: none found. Any more: classification error; output none
        print( 'class: none')
    #cv2.imshow( 'class', img)

### END-CLASS CLASSIFICATION

### START-CHARACTER CLASSIFICATION

def findCharacters( rows):
    ## Template file
    templateFile = cv2.imread( './Template/AtoZFinal.PNG', cv2.IMREAD_COLOR)
    ## Ensure a character map in the following order
    charKey = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', \
                'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', \
                    'X', 'Y', 'Z']
    
    ## Use the center third of the image for text detection
    img = rows[1]
    ## Convert to gray scale and apply pre-processing to prepare contour detection
    gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    ## Make a copy of the original with contours not drawn on
    gray_ori = gray_img.copy()
    
    #cv2.imshow('row2',gray_ori)
                               
    ## Apply a Gaussian blur to the gray image to eliminate uninteresting stable regions
    gray_img = cv2.GaussianBlur( gray_img, (5, 5), 0)
    
    ## Perform MSER to find character symbols (regions of interest)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions( gray_img)
    
    ## Find the contours using convexHull
    hulls = [ cv2.convexHull( p.reshape( -1, 1, 2)) for p in regions ]
    
    ## Initialise mask for text area
    mask = np.zeros( gray_img.shape, np.uint8)
    
    for cnts in hulls:
        per = cv2.arcLength( cnts, True)
        aprx = cv2.approxPolyDP( cnts, 0.01 * per, True)
        
        ## Eliminate contours that are too small or too large
        if cv2.contourArea( aprx) > 250.0 and cv2.contourArea( aprx) < 3200:
            ( x, y, w, h) = cv2.boundingRect( cnts)
            ## If not touching borders and roughly in the center:
            if not (x == 1 or y == 1):
                t1 = x + w
                t2 = y + h
                if not ( t1 == gray_img.shape[1]-1 or t2 == gray_img.shape[0]):
                    if not y < gray_img.shape[0]-120:
                        ## Set the mask
                        cv2.rectangle( mask, ( x, y), ( x+w, y+h), (255, 255, 255), -1)
    
    ## Dilate the mask to find the text bounding box
    mask = cv2.dilate( mask, np.ones( ( 5, 5), np.uint8), iterations=2)
    result = cv2.bitwise_and( gray_img, mask)
    
    #cv2.imshow( 'contours', result)
    
    ## Initialise some fields:
    ## @black = colour for padding
    ## @pad = border padding
    ## @textout = final text checking string
    ## @resultText = the output to the user
    ## @output[] = output list where all characters and words joined
    black = [ 0, 0, 0]
    pad = 20
    textout = ''
    resultText = 'text: '
    output = []
    
    ## Mask needs to be appropriate for words
    m_cnt = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_cnt = m_cnt[1]
    
    ## Only find the bounding box it makes sense to detect the word
    for c in m_cnt:
        ( x, y, w, h) = cv2.boundingRect( c)
        if (w > 50 and h > 30 and h < 70 ) or (w > 150 and h > 70):
            cv2.rectangle( gray_img, ( x, y), ( x+w, y+h), (255, 255, 255), 1)
            
            ## ROI where we calculate them to be (from above)
            text_roi = gray_ori[ y : y+h, x : x+w]
            #cv2.imshow( 'text_roi', text_roi)
            
            ## Call function to detect the individual letters in the template stored in a dictionary
            charas = {}
            charas = findContourOCR( templateFile)
            
            ## Filter the text_roi
            text_roi = cv2.medianBlur( text_roi, 3)
            
            ## Apply adaptive thresholding to get the characters segmented:
            text_roi = cv2.adaptiveThreshold( text_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
            
            ## If needed; dilate erosion filters
            #text_roi = cv2.dilate( text_roi, np.ones( ( 5, 5), np.uint8), iterations=1)
            #text_roi = cv2.erode( text_roi, np.ones( ( 3, 3), np.uint8), iterations=1)
            
            ## If image is more white than black, invet, else leave as is
            num_white = np.sum( text_roi == 255)
            #print( num_white)
            ## Check white pixel density to work with as many labels as possible
            if num_white < 1500 or num_white > 1800 and num_white < 4200 or num_white > 5800 and num_white < 6000 or num_white > 12000: 
                text_roi = cv2.bitwise_not(text_roi)
            
            ## Pad the borders to ensure all text are inside the contour
            text_roi = cv2.copyMakeBorder( text_roi.copy(), pad/2, pad/2, pad/2, pad/2, cv2.BORDER_CONSTANT, value=black)
            
            ## Morphology and blurs to clear up the text 
            text_roi = cv2.medianBlur( text_roi, 5)
            text_roi = cv2.erode( text_roi, np.ones( ( 3, 3), np.uint8), iterations=1)
            
            text_roi = cv2.dilate( text_roi, np.ones( ( 5, 5), np.uint8), iterations=1)
            #text_roi = cv2.erode( text_roi, np.ones( ( 5, 5), np.uint8), iterations=1)
            #cv2.imshow( 'filtered text_roi', text_roi)
            
            ## Find contours of the text area and template match
            locs = findContourEachChar( text_roi)
            
            ## Output will show the extracted number
            groupOutput = []
            
            ## Loop over the existing potential numbers:
            for( i, ( lX, lY, lW, lH)) in enumerate( locs):
                
                ## Extract the char from binary section image
                roi = text_roi[ lY: lY + lH+1, lX: lX + lW+1]
                
                ## Ensure same calculated roi and actual roi - erosion above aswell
                roi = cv2.erode( roi, np.ones( ( 3, 3), np.uint8), iterations=1)
                #cv2.imshow( 'calculated roi', roi)
                
                ## Initialise a list of template matching scores
                scores = []
                
                if roi.shape[0] and roi.shape[1] > 0:
                    ## Ensure ROI is the same size as the template thresholding for scoring
                    roi = cv2.resize( roi, ( 57, 88))
                    ## Pad the ROI to help out the template matching
                    roi = cv2.copyMakeBorder( roi.copy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=black)
                    ## If the white area is too great, invert the image to score against template
                    
                    roi_white = np.sum( roi == 255)
                    if roi_white < 700:
                        roi = cv2.bitwise_not(roi)
                    #cv2.imshow( 'actual roi', roi)
                    
                    ## Loop over reference char and char ROI:
                    for( char, charROI) in charas.items():
                        ## Apply template matching and score on max value of match
                        result = cv2.matchTemplate( roi, charROI, cv2.TM_CCOEFF)
                        ( _, score, _, _) = cv2.minMaxLoc( result)
                        scores.append( score)
                        
                    ## Scores should match with template contours -> already sorted     
                    groupOutput.append( str( charKey[ np.argmax( scores)]))
                    
            ## Update the digits output list
            output.extend( groupOutput)
            if len( groupOutput) > 0:
                ## Join up the characters
                bufferText = ''.join( groupOutput)
            
    ## SPELL CHECK
    if len( output) > 0:
        corrected = ''
        ## Join up any separated words/characters 
        textout += ''.join( output)
        
        if 'ACT' in textout or 'TIV' in textout or 'RAD' in textout:
            corrected = 'RADIOACTIVE'
        if 'IXF' in textout or 'EXP' in textout or 'FLOS' in textout or 'PLDS' in textout or 'PLOS' in textout:
            corrected = 'EXPLOSIVE'
            if 'S' in textout[1:]: ##End of string
                corrected = 'EXPLOSIVES'
        if 'AST' in textout or 'LAST' in textout or ('AG' in textout):
            corrected = 'BLASTING AGENT'
            if 'S' in textout[1:] or 'G' in textout[1:]: ##End of string
                corrected = 'BLASTING AGENTS'
        if 'P' in textout[1:] and 'I' in textout or 'ROX' in textout or 'GAN' in textout:
            corrected = 'ORGANIC PEROXIDE'
        if ('DR' in textout or 'OR' in textout) and 'I' in textout:
            corrected = 'ORGANIC PEROXIDE'
        if 'INH' in textout or 'H' in textout and 'AZ' in textout:
            if len(textout) > 7:
                corrected = 'INHALAZTION HAZARD'
        if 'T' in textout and 'X' in textout and len( textout) < 8:
            corrected = 'TOXIC'
        if 'X' in textout and 'GE' in textout:
            corrected = 'OXYGEN'
        if 'MMA' in textout or 'GMA' in textout or 'ABL'in textout and 'AS' in textout:
            corrected = 'FLAMMABLE GAS'
        if 'AMM' in textout and not 'G' in textout:
            corrected = 'FLAMMABLE'
        if 'DA' in textout and 'AN' in textout or 'DAN' in textout:
            corrected = 'DANGEROUS WHEN WET'
        if 'BUST' in textout:
            corrected = 'COMBUSTIBLE'
        if 'UE' in textout:
            corrected = 'FUEL OIL'
        if 'DXID' in textout or 'OXID' in textout:
            if len( textout) < 9:
                corrected = 'OXIDIZER'
        if 'ISON' in textout or 'ISDN' in textout:
            corrected = 'POISON'
        if 'COR' in textout or 'CDR' in textout and 'IV' in textout:
            corrected = 'CORROSIVE'
        if 'GASOL' in textout or 'GASUL' in textout:
            corrected = 'GASOLINE'
        print( resultText + corrected)
    else:
        print( 'text: (none)')
    
### END-CHARACTER CLASSIFICATION
