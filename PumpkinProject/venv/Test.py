import cv2
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input = "../Input/images/"
output = "../Output/images/"

# Load the image into picture matrix#
picture = cv2.imread(input+"DJI_0237.JPG")

# function to show image.
def showimg(imgname, image, x=None,y=None):
    cv2.namedWindow(imgname, cv2.WINDOW_NORMAL)
    if x is not None or y is not None:
        cv2.resizeWindow(imgname, x, y)
    cv2.imshow(imgname, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
# Reference image: Original_image; image to be searched by using referenceimg,
def backProj(referenceimg, original_image):
    ref_im_in_hsv = cv2.cvtColor(referenceimg, cv2.COLOR_BGR2HSV) # convert input reference image to HSV
    HSV = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV) # convert input search image to hsv
    # calculate histogram from hsv pumpkin
    ref_im_histogram = cv2.calcHist(images=[ref_im_in_hsv],channels=[0,1], mask=None, histSize=[180 , 256], ranges=[0,180,0,256])
    # normalize the histogram
    cv2.normalize(src=ref_im_histogram,dst=ref_im_histogram,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    # backproject thbackproject_imge image without any filters or manipulations
    backproject_img = cv2.calcBackProject(images=[HSV],channels=[0, 1],hist=ref_im_histogram,ranges=[0, 180, 0, 256],scale=1)
    # do a Dilation
    disc_se = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))
    backproject_img = cv2.filter2D(src=backproject_img,ddepth=-1,kernel=disc_se)


    # do the last of the overlay, from the backproject to original image.
    ret, bp_threshold = cv2.threshold(src=backproject_img,thresh=50,maxval=255,type=cv2.THRESH_BINARY)
    bp_threshold = cv2.merge((bp_threshold,bp_threshold,bp_threshold))
    res = cv2.bitwise_and(original_image, bp_threshold)
    stack = np.vstack((original_image, bp_threshold, res))
    #showimg("stack", stack,600,1024)
    return stack
# prints mean and standard deviation of either a color or B/W image
def find_info(image, Colorimage):
    if Colorimage == True:
        Colors = ["Blue", "Green", "Red"]
        for i in range (0,3):
            print Colors[i]
            print "Mean: " + str(np.mean(image[:,:,i]))
            print "Standard deviation: " + str(np.std(image[:,:,i])) + "\n"
    else:
        print "Mean: " + std(np.mean(image))
        print "Standard deviation: " + std(np.std(image)) + "\n"

def countPumpkins(binaryimage):
    contours, hierarchy = cv2.findContours(binaryimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    A =  []
    for i in range (0, np.size(contours)):
        area = cv2.contourArea(contours[i])
        if area > 0:
            A.append(contours[i])

    print "\n Number of blobs in the image is: " + str(np.size(A))

def MarkPumpkins(org_img, binaryimg):
    contours, hierarchy = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(org_img, contours, -1, (0,0,255),3)
    showimg("Marked pumpkins",org_img)

def savePicture(imagename, image):
    cv2.imwrite(output+imagename+'.jpg', image)

#Get reference pumpkins
height= 12
Pumpkin = picture[2193:2193+height, 2395:2395+height] # Reference image
pumpkins = Pumpkin
Pumpkin = picture[1406:1406+height, 2598:2598+height] # Reference image
pumpkins = np.hstack((Pumpkin,pumpkins))
Pumpkin = picture[1125:1125+height, 2715:2715+height] # Reference image
pumpkins = np.hstack((Pumpkin,pumpkins))
savePicture("Pumpkins", pumpkins)

# exercise 1: find mean and standard deviations of the pumpkins.
print "Find statistics of the RGB image; e.g mean and standard deviation"
find_info(pumpkins, True)

# segment the image from CieLab values.
cieLab_ref = cv2.cvtColor(pumpkins, cv2.COLOR_BGR2LAB)
print "\nFinding CieLAB image statistics"
find_info(cieLab_ref, True)

# Find pumpkins from thresholds.
mask = cv2.inRange(picture, (30-(15*2),94-(22*2),170-(20*2)),(30+(15*2),94+(22*2),170+(20*2)))
kernel = np.ones((9,9),np.uint8)
mask1 = cv2.dilate(mask, kernel, iterations=1)
#showimg("Finding blobs from RGB space", mask1)
savePicture("RGB_Threshold", mask1)

#Do a CieLAB color thresholding.
cieLab_image = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB)
std_dev = 7
binaryimg = cv2.inRange(cieLab_image, (122-(20*std_dev),154-(4*std_dev),175-(4*std_dev)),(122+(20*std_dev),154+(4*std_dev),175+(4*std_dev)))
kernel = np.ones((9,9),np.uint8)
binaryimg = cv2.morphologyEx(binaryimg, cv2.MORPH_OPEN, kernel)
MarkPumpkins(picture, binaryimg)
countPumpkins(binaryimg)

#Stack images up uppon, so we can compare
ret, bp_threshold = cv2.threshold(src=binaryimg,thresh=50,maxval=255,type=cv2.THRESH_BINARY)
bp_threshold = cv2.merge((bp_threshold,bp_threshold,bp_threshold))
res = cv2.bitwise_and(picture, bp_threshold)
stack = np.vstack((picture, bp_threshold, res))
savePicture("cieLab_8", stack)
#Show the CieLAB color thresholding image.
#showimg("CieLAB thresholding", stack)

# exercise 2: Segment the image from different methods
# use backprojection segmentation to find pumpkins.
pumpkinsimage = backProj(pumpkins, picture)
#showimg("Backprojected image", pumpkinsimage)
savePicture("Backprojected_image", pumpkinsimage)

height, width, channels = pumpkins.shape
Blue,Green,Red= cv2.split(pumpkins)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
Red_mean = np.mean(Red)
Green_mean = np.mean(Green)
Blue_mean = np.mean(Blue)
ax.scatter(Red_mean,Green_mean,Blue_mean, c='g', s=100)
ax.scatter(Red,Green,Blue, c='r')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_zlim(0,255)
plt.show()

bgr = [Blue_mean,Green_mean,Red_mean]
threshold = 50

minBGR = np.array([bgr[0]-threshold, bgr[1]-threshold, bgr[2]-threshold])
maxBGR = np.array([bgr[0]+threshold, bgr[1]+threshold, bgr[2]+threshold])
binaryimg_RGB_distance = cv2.inRange(picture, minBGR, maxBGR)
binaryimg_RGB_distance = cv2.merge((binaryimg_RGB_distance,binaryimg_RGB_distance,binaryimg_RGB_distance))
result = cv2.bitwise_and(binaryimg_RGB_distance, picture)

print math.sqrt(54**2+30**2)

#showimg("bgr distance", result)
