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
def nothing( x):
    pass

def detectHSVColours( img):
    #img = cv2.imread( 'P1380454.JPG', 1)
    blurred = cv2.GaussianBlur( img, (5,5), 0)
    #closed = cv2.cvtColor( blurred, cv2.COLOR_GRAY2BGR)
    
    closed = cv2.morphologyEx( blurred, cv2.MORPH_OPEN, np.ones( ( 5, 5), np.uint8))
    
    hsv_img = cv2.cvtColor( closed, cv2.COLOR_BGR2HSV)
    
    cv2.imshow( 'hsv', hsv_img)
    
    colour1 = hsv_img[ 190, 395]
    colour2 = hsv_img[ 450, 250]
    print( colour1)
    print( colour2)
    
    topShow = 'top:'
    botShow = 'bot:'
    
    #Orange, Red, Yellow need a tighter range check in HSV than the others as they are similar
    
    ## Check for Green
    if colour1[0] >= 20 and colour1[0] <= 90:
        if colour1[1] >= 70 and colour1[1] <= 140:
            topShow += ' green'
        
    if colour2[0] >= 20 and colour2[0] <= 90:
        if colour2[1] >= 70 and colour2[1] <= 140:
            botShow += ' green'
            
    ## Check for Orange
    if (colour1[0] >= 0 and colour1[0] <= 16) or (colour1[0] >= 176 and colour1[0] <= 179):
        if (colour1[1] >= 171 and colour1[1] <= 255) or (colour1[2] >= 195 and colour1[2] <= 215):
            topShow += ' orange'
        
    if (colour2[0] >= 0 and colour2[0] <= 16) or (colour2[0] >= 176 and colour2[0] <= 179):
        if (colour2[1] >= 171 and colour2[1] <= 255) or (colour2[2] >= 75 and colour2[2] <= 90):
            botShow += ' orange'
    
    ## Check for Red
    if (colour1[0] >= 160 and colour1[0] <= 179) or (colour1[0] >= 0 and colour1[0] <= 10):
        if colour1[1] >= 100 and colour1[1] <= 180:
            topShow += ' red'
        
    if (colour2[0] >= 160 and colour2[0] <= 179) or (colour2[0] >= 0 and colour2[0] <= 10):
        if colour2[1] >= 100 and colour2[1] <= 180:
            botShow += ' red'
    
    ## Check for Blue
    if colour1[0] >= 105 and colour1[0] <= 125:
        if colour1[2] >= 108 and colour1[2] <= 150:
            if (colour1[1] >= 35 and colour1[1] <= 90) or (colour1[1] >= 135 and colour1[1] <= 200):
                topShow += ' blue'
    
    if colour2[0] >= 105 and colour2[0] <= 125:
        if colour2[2] >= 108 and colour2[2] <= 150:
            if (colour2[1] >= 35 and colour2[1] <= 90) or (colour2[1] >= 135 and colour2[1] <= 200):
                botShow += ' blue'
    
    ## Check for White
    if (colour1[0] >= 0 and colour1[0] <= 120):
        if (colour1[1] >= 0 and colour1[1] <= 90):
            if colour1[2] >= 140:
                topShow += ' white'
        
    if (colour2[0] >= 0 and colour2[0] <= 120):
        if (colour2[1] >= 0 and colour2[1] <= 90):
            if colour2[2] >= 140:
                botShow += ' white'
        
    ## Check for Black
    if colour1[0] >= 10 and colour1[0] <= 50:
        if colour2[2] >= 0 and colour2[2] <= 105:
            topShow += ' black'
    
    if colour2[0] >= 10 and colour2[0] <= 50:
        if colour2[2] >= 0 and colour2[2] <= 105:
            botShow += ' black'
            
    ## Check for Yellow
    if colour1[0] >= 17 and colour1[0] <= 30:
        if colour1[1] >= 168 and colour1[1] <= 214:
            topShow += ' yellow'
            
    if colour2[0] >= 17 and colour2[0] <= 30:
        if colour2[1] >= 168 and colour2[1] <= 214:
            botShow += ' yellow'
    
    
    print( topShow)
    print( botShow)
    
def findColours( img):
    ## Define the lower and upper boundaries for expected colours
    ## White  - Lower: H:0 S:0 V:230    | Upper: H: 255 S:0 V:255  
    ## Black  - Lower: H:0 S:0 V:0      | Upper: H:255 S:255 V:0
    ## Yellow - Lower: H:23 S:60 V:120  | Upper: H:54 S:255 V:255
    ## Red    - Lower: H:165 S:204 V:128| Upper: H:186 S:230 V:255
    ## Orange - Lower: H:10 S:50 V:80   | Upper: H:20 S:255 V:255
    ## Green  - Lower: H:60 S:120 V:129 | Upper: H:83 S:255 V:255
    ## Blue   - Lower: H:98 S:110 V:100 | Upper: H:117 S:255 V:255

    ## Lower key-value
    lower = { 'white':( 0, 0, 0), #
              'black':( 0, 0, 0), #
              'yellow':( 22, 180, 230), #
              'red':( 160, 0, 180),#
              'orange': ( 0, 112, 207), #
              'green':( 38, 94, 0), #
              'blue':( 75, 60, 26)}
    
    upper = { 'white':( 0, 0, 178), #
              'black':( 0, 0, 76), #
              'yellow':( 38, 255, 255), #
              'red':( 179, 0, 255), #
              'orange': ( 22, 138, 255), #
              'green':( 75, 255, 0), #
              'blue':( 130, 116, 255)}
    
    blurred = cv2.GaussianBlur( img, (5,5), 0)
    #closed = cv2.cvtColor( blurred, cv2.COLOR_GRAY2BGR)
    
    closed = cv2.morphologyEx( blurred, cv2.MORPH_OPEN, np.ones( ( 5, 5), np.uint8))
    cv2.imshow( 'colour morph', closed)
    
    colour1 = closed[190,395]
    ## If colour = white, black, yellow, red, orange, green, blue... BGR
    colour2 = closed[450,250]
    print( colour1)
    print( colour2)
    
    topHalf = 'top: '
    bottomHalf = 'bottom: '
    
    ## Check for Black
    if colour1[0] >= 0 and colour1[0] <= 90:
        if colour1[1] >= 0 and colour1[1] <= 90: 
            if colour1[2] >= 0 and colour1[2] <= 90:
                topHalf += 'black'
                #rint( topHalf)
        
    if colour2[0] >= 0 and colour2[0] <= 90:
        if colour2[1] >= 0 and colour2[1] <= 90:
            if colour2[2] >= 0 and colour2[2] <= 90:
                bottomHalf += 'black'
                #print( bottomHalf)
                
    ## Check for White
    if colour1[0] >= 105 and colour1[0] <= 255:
        if colour1[1] >= 105 and colour1[1] <= 255:
            if colour1[2] >= 110 and colour1[2] <= 255:
                topHalf += 'white'
                #print( topHalf)
    
    if colour2[0] >= 105 and colour2[0] <= 255:
        if colour2[1] >= 105 and colour2[1] <= 255:
            if colour2[2] >= 110 and colour2[2] <= 255:
                bottomHalf += 'white'
                #print( bottomHalf)
                
    ## Check for Yellow
    if colour1[0] >= 0 and colour1[0] <= 65:
        if colour1[1] >= 80 and colour1[1] <= 199:
            if colour1[2] >= 130 and colour1[2] <= 255:
                topHalf += 'yellow'
                #print( topHalf)
    
    if colour2[0] >= 0 and colour2[0] <= 65:
        if colour2[1] >= 80 and colour2[1] <= 199:
            if colour2[2] >= 130 and colour2[2] <= 255:
                bottomHalf += 'yellow'
                #print( bottomHalf)
                
    ## Check for Red
    if colour1[0] >= 15 and colour1[0] <= 85:
        if colour1[1] >= 0 and colour1[1] <= 80:
            if colour1[2] >= 110 and colour1[2] <= 255:
                topHalf += 'red'
                #print( topHalf)
    
    if colour2[0] >= 15 and colour2[0] <= 85:
        if colour2[1] >= 0 and colour2[1] <= 80:
            if colour2[2] >= 110 and colour2[2] <= 255:
                bottomHalf += 'red'
                #print( bottomHalf)
                
    ## Check for Orange
    if colour1[0] >= 55 and colour1[0] <= 70:
        if colour1[1] >= 80 and colour1[1] <= 145:
            if colour1[2] >= 200 and colour1[2] <= 255:
                topHalf += 'orange'
                #print( topHalf)
    
    if colour2[0] >= 55 and colour2[0] <= 70:
        if colour2[1] >= 80 and colour2[1] <= 145:
            if colour2[2] >= 200 and colour2[2] <= 255:
                bottomHalf += 'orange'
                #print( bottomHalf)
                
    ## Check for Green
    if colour1[0] >= 45 and colour1[0] <= 65:
        if colour1[1] >= 125 and colour1[1] <= 255:
            if colour1[2] >= 0 and colour1[2] <= 180:
                topHalf += 'green'
                #print( topHalf)
    
    if colour2[0] >= 45 and colour2[0] <= 65:
        if colour2[1] >= 125 and colour2[1] <= 255:
            if colour2[2] >= 0 and colour2[2] <= 180:
                bottomHalf += 'green'
                #print( bottomHalf)
    
    ## Check for Blue
    if colour1[0] >= 110 and colour1[0] <= 255:
        if colour1[1] >= 55 and colour1[1] <= 110:
            if colour1[2] >= 30 and colour1[2] <= 110:
                topHalf += 'blue'
                #print( topHalf)
    
    if colour2[0] >= 110 and colour2[0] <= 255:
        if colour2[1] >= 55 and colour2[1] <= 110:
            if colour2[2] >= 30 and colour2[2] <= 110:
                bottomHalf += 'blue'
                #print( bottomHalf)
                
    print( topHalf)
    print( bottomHalf)
    
    ## Pre-process: 
    ## Apply a GaussianBlur
    #gblur = cv2.GaussianBlur( img, ( 5, 5), 0)
    
    ## Convert image into HSV 
    #rgb = cv2.cvtColor( gblur, cv2.COLOR_BGR2RGB)
    
    ## For each colour in pre-defined colour values
    #for key, value in upper.items():
        ## Construct a mask for the colour of each key
        ## Supply an erosion and dilation kernel
        #kern = np.ones( ( 9, 9), np.uint8)
        
        #mask = cv2.inRange( img, lower[key], upper[key])
        #cv2.imshow( 'out', mask)
        #print( lower[key], ' ',upper[key])
        
        ## Perform some erosions and dilations to close up any noise
        #mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kern)
        #mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kern)
        
        ## Find the contours of the mask
        #cnts = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[ -2]
        
        #for c in cnts:
            #perimeter = cv2.arcLength( c, True)
            #aprx = cv2.approxPolyDP( c, 0.02 * perimeter, True)
            
            #if cv2.contourArea( aprx) > float( 0.5):
                #mask = cv2.drawContours( mask, [aprx], -1, (0,0,255), 3)
    return closed
    
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