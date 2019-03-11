import cv2
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input = "/Input/images/"
output = "/Output/images/"

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
        Colors = ["Channel 1", "Channel 2", "Channel 3"]
        for i in range (0,3):
            print Colors[i]
            print "Mean: " + str(np.mean(image[:,:,i]))
            print "Standard deviation: " + str(np.std(image[:,:,i])) + "\n"
    else:
        print "Mean: " + str(np.mean(image))
        print "Standard deviation: " + str(np.std(image)) + "\n"

def countPumpkins(binaryimage):
    contours, hierarchy = cv2.findContours(binaryimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print "Number of contours found in the image. : " + str(np.size(contours))

def MarkPumpkins(name, org_img, binaryimg):
    contours, hierarchy = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(org_img, contours, -1, (0,0,255),2)
    showimg(name,org_img)

def savePicture(imagename, image):
    cv2.imwrite(output+imagename+'.jpg', image)

def RGB_space3D(image):
    #Show RGB space of the pumpkins array
    Blue, Green, Red= cv2.split(image)
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

#Get reference pumpkins
height= 12
Pumpkin = picture[2193:2193+height, 2395:2395+height] # Reference image
pumpkins = Pumpkin
Pumpkin = picture[1406:1406+height, 2598:2598+height] # Reference image
pumpkins = np.hstack((Pumpkin,pumpkins))
Pumpkin = picture[1125:1125+height, 2715:2715+height] # Reference image
Ref_pumpkins = np.hstack((Pumpkin,pumpkins)) #Stack those images beside each other.
savePicture("Pumpkins", pumpkins)

# exercise 1: find mean and standard deviations of the pumpkins.
print "Find statistics of the RGB image; e.g mean and standard deviation"
find_info(Ref_pumpkins, True)   #Print mean and standard deviations

# Find mean and standard deviation from CieLab values.
cieLab_ref = cv2.cvtColor(Ref_pumpkins, cv2.COLOR_BGR2LAB)  #Convert to LAB image space
print "\nFinding CieLAB image statistics"
find_info(cieLab_ref, True)     #Print mean and standard deviations


# exercise 2: Segment the image from different methods
# Segment pumpkins from RGB thresholds.
std_dev = 2
mask = cv2.inRange(picture, (30-(15*std_dev),94-(22*std_dev),170-(20*std_dev)),(30+(15*std_dev),94+(22*std_dev),170+(20*std_dev)))
# Show and save the result.
showimg("Finding blobs from RGB space", mask)
savePicture("RGB_Segmented", mask)

#Do a CieLAB color segmented image by thresholding.
cieLab_image = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB) #Convert whole image to LAB image space
std_dev = 8
cielab_segmented = cv2.inRange(cieLab_image, (122-(20*15),154-(4*std_dev),175-(4*std_dev)),(122+(20*15),154+(4*std_dev),175+(4*std_dev)))
savePicture("LAB_Segmented", cielab_segmented)
showimg("LAB_Segmented", cielab_segmented)

# use backprojection segmentation to find pumpkins.
pumpkinsimage = backProj(Ref_pumpkins, picture)
showimg("Backprojected image", pumpkinsimage)
savePicture("Backprojected_image", pumpkinsimage)

#RGB Space distance thresholding.
RGB_space3D(pumpkins)
RGB_Space = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)


#Does not work to make the RGB distance calculation :(
A = np.arange(3078*5472).reshape(3078,5472)
A[:,:] = np.sqrt(((30-picture[:,:,0])**2)+((94-picture[:,:,1])**2)+((170-picture[:,:,2])**2))

'''
Uncomment this part for a seriously slow "Distance in RGB space" function
for i in range (0, 3078):
    for j in range (0, 5472):
        if (np.sqrt(((30-picture[i,j,0])**2)+((94-picture[i,j,1])**2)+((170-picture[i,j,2])**2)) > 255, ):
            RGB_Space = 255
        else:
            RGB_Space[i,j] = np.sqrt(((30-picture[i,j,0])**2)+((94-picture[i,j,1])**2)+((170-picture[i,j,2])**2))
showimg("A", RGB_Space)
savePicture("RGB_Distance", RGB_Space)
'''

# Exercise 3

# Find number of pumpkins in the LAB segmented image.
print "cieLAB count pumpkins without filtering."
countPumpkins(cielab_segmented)
cielab_marked = picture.copy()
MarkPumpkins("LAB image marked", cielab_marked, cielab_segmented)
savePicture("LAB_Marked", cielab_marked)

# Exercise 4 and 5 and 7

# Filter the segmented image and save.
print "\nFiltering the image and counting pumpkins \n"
kernel = np.ones((5,5),np.uint8)
cielab_segmented = cv2.morphologyEx(cielab_segmented, cv2.MORPH_OPEN, kernel)
cielab_segmented_filtered = cv2.medianBlur(cielab_segmented, 7)
LAB_filtered_marked = picture.copy()
countPumpkins(cielab_segmented_filtered)
MarkPumpkins("LAB filtered marked", LAB_filtered_marked,cielab_segmented_filtered)
savePicture("LAB_filtered_marked", LAB_filtered_marked)
