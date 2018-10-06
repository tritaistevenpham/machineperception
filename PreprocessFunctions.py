## Import libraries
import cv2
import numpy as np

## Import user-defined functions
import TransformFunctions as tf
import pytesseract as pt

### START-MASK DETECTION

def preprocessImage( img):
    ##  First convert the image into grayscale for pre-processing
    warp = img.copy()
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
            
            warp = tf.fourPtTrans( res, points)
        
    return warp

### END-MASK DETECTION

### START-PREPROCESSING FOR COLOUR FUNCTION

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
    cv2.imshow( 'colour balance', ret_img)
    
    return ret_img

def detectHSVColours( img):
    
    ## Give the image a blur to filter out small noise
    blurred = cv2.GaussianBlur( img, (5,5), 3)
    
    ## Help the smoothing with a morphology erosion and dilation
    closed = cv2.morphologyEx( img, cv2.MORPH_OPEN, np.ones( ( 5, 5), np.uint8))
    
    ## Use the HSV colourspace to classify colours
    hsv_img = cv2.cvtColor( closed, cv2.COLOR_BGR2HSV)
    
    ## Select likely top and bottom of the image
    colour1 = hsv_img[ 190, 395]
    colour2 = hsv_img[ 370, 190]
    print( colour1)
    print( colour2)
    
    ## Initialise colour output
    topShow = 'top:'
    botShow = 'bot:'
    actualTop = ''
    actualBot = ''
    
    ## Colour HSV range thresholding to find the correct values to classify the data set based
    ## on Simple Background Images
    
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

def divideImage( img_orig):
    ## Set the img width and height
    img = img_orig.copy()
    imgh, imgw = img.shape[ :2]
    
    numRows = 5
    
    ## Set value to divide the image up: 5 equal rows
    rowHeight = imgh / 5.0
    
    ## Draw the mask for the number of sections -> 5
    #for ii in range( 0, numRows):
        

### END-DIVIDE IMAGE FUNCTION

### START-OCR

def readSign( img):
    
    ## List of results:
    results = []
    text_colon = 'text: '
    text_word = 'word: '
    ## MSER
    vis = img.copy()
    mser = cv2.MSER_create()
    
    gray = cv2.cvtColor( img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.dilate( gray, np.ones( ( 3, 3), np.uint8), iterations = 1)
    
    edges = cv2.Canny( gray, 10, 50)
    
    contours = cv2.findContours( edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    #showCont = cv2.drawContours( img, cnt, -1, (0, 255, 0), 1)
    
    regions, _ = mser.detectRegions( gray)
    hulls = [ cv2.convexHull( p.reshape( -1, 1, 2)) for p in regions]
    
    ## Set up mask
    mask = np.zeros( img.shape, np.uint8)
    characters = []
    
    idx = 0
    segm = './Segments/'
    words = []
    
    ## Find the contours of region greater than 40, set a bounding box
    for c in regions:
        perimeter = cv2.arcLength( c, True)
        aprx = cv2.approxPolyDP( c, 0.01 * perimeter, True)
        
        if cv2.contourArea( aprx) > 50.0 and cv2.contourArea( aprx) < 1000:
            x, y, w, h = cv2.boundingRect( c)
            cv2.rectangle( mask, ( x, y), ( x+w, y+h), ( 0, 255, 0), 1)
            ## Store the coordinates into an object, use for tesseract:
            characters.append( c)
            
    ## Loop over all the characters:
    k1 = np.ones( ( 5, 5), np.uint8)
    pad = 20 #px
    color = [0, 0, 0]

    """
    for c in characters:
        (x, y, w, h) = cv2.boundingRect( c)
        x -= 1
        y -= 1
        w += 2
        h += 2
        
        roi = img[ y: y+h, x: x+w]
        
        roi_re = cv2.resize( roi, (150, 150))
        
        roi_re = cv2.copyMakeBorder( roi_re, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)
        
        gray_roi = cv2.cvtColor( roi_re.copy(), cv2.COLOR_BGR2GRAY)
        
        ret, bin_roi = cv2.threshold( gray_roi, 127, 255, cv2.THRESH_BINARY)
        
        bin_roi = cv2.bitwise_not( bin_roi)
        #roi = cv2.threshold( gray_roi, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bin_roi = cv2.dilate( bin_roi, k1, iterations=3)
        
        cv2.imwrite( (segm + str(idx) + '.png'), bin_roi)
        
        cv2.imshow( 'c', bin_roi)
        idx += 1
        
        single_char = pt.image_to_string( bin_roi, lang='eng', boxes=False, config='--psm 10 --oem 3 -c \
                                         tessedit_char_whitelist=ABCDEFGHIKLMNOPQRSTUVWXYZ')
        words.append( single_char)
    #    #cv2.drawContours( mask, [aprx], -1, ( 0, 255, 255), 1)
    
    print( words)
    for ii in words:
        ii = "".join( [ c if ord( c) < 128 else "" for c in ii]).strip()
        text_word += ii
    print( text_word)
    cv2.imshow( 'mask', mask)"""
    
    #img_blur = cv2.GaussianBlur( img, ( 5, 5), 0)
    
    kernel = np.ones( ( 3, 3), np.uint8)
    
    img_rgb = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.Canny( img_gray, 100, 200)
    
    #img_gray = cv2.dilate( img_gray, kernel, iterations=1)
    
    cv2.imshow( 'gray', img_gray)
    inv = cv2.bitwise_not( img_gray)
    cv2.imshow( 'inv', inv)
    
    #closed = cv2.morphologyEx( inv, cv2.MORPH_CLOSE, kernel)
    """
    new_img = np.zeros_like( img_gray)
    for val in np.unique( img_gray)[ 1:]:
        new_mask = np.uint8( img_gray == val)
        output = cv2.connectedComponentsWithStats( new_mask, 4)[ 1:3]
        labels = output[0]
        stats = output[1]
        
        largest_label = 1 + np.argmax( stats[ 1:, cv2.CC_STAT_AREA])
        new_img[ labels == largest_label] = val
        print( labels)
        
    print( new_img)
    cv2.imshow( 'connected', new_img)
    """
    
    text = pt.image_to_string( inv, lang='eng', boxes=False, config='--psm 7 --oem 3 -c \
                                         tessedit_char_whitelist=ABCDEFGHIKLMNOPQRSTUVWXYZ012345678')
    
    ## Append detected text to list
    results.append(text) 
    
    ## Strip out non-ascii text
    for text in results:
        text = "".join( [ c if ord( c) < 128 else "" for c in text]).strip()
        text_colon += text
    print( text_colon)
    #print( results)
    
### END-OCR
