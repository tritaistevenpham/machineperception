## COMP3007 - Machine Perception Assignment
## Author: Tri Tai Steven Pham
## Student ID: 17748229
## Year: 2018

## Import libraries
import cv2
import numpy as np

## Main
fileLoc = ''
option = 0
## Choose: 1 - Read a single file from SetA
##         2 - Read all images from a directory

while option != 1 and option != 2:
    try: option = int(raw_input( 'Enter either 1 or 2\n 1 - Read a single file from SetA\n 2 - Read all images from directory\n'))
    except ValueError:
        print('Please only use integers')

if option == 1: ## 4 + shadow P1380524.JPG | 2 P1380513.JPG | 1 P1380502.JPG 
    fileLoc = './P1380502.JPG'
elif option == 2:
    fileLoc = './20180826a/SetA/label-7-radioactive-ii.png'

## Option 1 + resize
img_orig = cv2.imread( fileLoc, cv2.IMREAD_COLOR)

r = 600.0 / img_orig.shape[1]
dim =  (600,  int( img_orig.shape[0] * r))

img = cv2.resize( img_orig, dim, interpolation = cv2.INTER_AREA)

## Option 2
## img = cv2.imread( dirLoc, cv2.IMREAD_COLOR)

## Color conversion - grayscale for image processing
gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

### SHAPE DETECTION
## Goal: Contour the diamond-shaped hazmat sign to be the main processing space.

##      Apply Gaussian blur with a 5 x 5 kernel to the image (Reduce high frequency noise)
gblur = cv2.GaussianBlur( gray_img, (5, 5), 0)

##      Compute the edge map
edge = cv2.Canny( gblur, 100, 255, 255)
#edge = cv2.Laplacian( gblur, cv2.CV_8UC1)

##      Find the contours within the computed edge map; sort by size in desc order
contours = cv2.findContours( edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
displayCnt = None

##      Loop over the contours
aprx = None
sign_arr = []
rect_coords = []
running_count = 0
for c in cnt:
    ## Approximate the contour
    al =  cv2.arcLength( c, True)
    aprx = cv2.approxPolyDP( c, 0.02 * al, True)

    div_four = len( aprx)
    
    # Draw the approximated contour for each corner 
    if div_four % 4 == 0:
        print('Approx coordinates: ', aprx)
        # Use bounding rectangle on the approximated points:
        rect = cv2.boundingRect( aprx)
        x, y, w, h = rect
        
        # Draw the rectangle on the set of 4 approximated values
        cv2.rectangle( img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.drawContours( img, aprx, -1, (0,255,0), 5)
        running_count = running_count + div_four
        
        # Append rect coords to an array of signs to store
        rect_coords = []
        rect_coords.append([x,y])
        rect_coords.append([x+w, y])
        rect_coords.append([x, y+h])
        rect_coords.append([x+w, y+w])
        sign_arr.append(rect_coords)
    else:
        # If approximation doesn't apply; sign obscured so not 100% certain
        break
    
    ## If the contour has 4 vertices, we have found "a" diamond shape.
    ##if len( aprx) == 4:
    ##    displayCnt = aprx
    ##    print('found ', div_four, ' vertices')
    ##    break

print('# Of Detected Vertices (of Signs): ', running_count)
num_signs = float(running_count) / 4.0
print('num_signs: ', num_signs)


##      Finding the four vertices means we can extract the contents
##      Extract the sign, and apply a PERSPECTIVE transform
##      So, for each sign in sign_arr; extract details.
M = None

for s in range(len(sign_arr)):
    pts1 = np.float32( [[ sign_arr[s][0][0], sign_arr[s][0][1]], 
                        [ sign_arr[s][1][0], sign_arr[s][1][1]], 
                        [ sign_arr[s][2][0], sign_arr[s][2][1]], 
                        [ sign_arr[s][3][0], sign_arr[s][3][1]]])
    
    pts2 = np.float32( [[0,0], [500,0], [0,500], [500, 500]])
    
    # Call the perspective transform function
    T = cv2.getPerspectiveTransform( pts1, pts2)
    
    # Warp perspective 
    warped = cv2.warpPerspective( img, T, (500, 500))
    
    cv2.imshow( 'Perspective Transform', warped)

### SHAPE DETECTION FINISH

## Show Images
cv2.imshow( 'Original Image', img)
cv2.imshow( 'Edge Map', edge)

## Destroy Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
