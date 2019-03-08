# coding=utf-8
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("remember to brake the car.mp4")

Color_values = [[],[],[],[],[],[],[],[],[],[]]

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if counter == 0:
        firstframe = frame
    if ret == True:
        temp_color = frame[90,553]
        Color_values[0].append(temp_color)
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        counter += 1
    else:
        break

#print Color_values[0][0][0]
'''
#Exercise 1
Locate five positions in the video (specify image coordinates) “remember to brake the car.mp4” which
are part of the background during the entire video.
Then locate five other positions in the video which
changes from background to foreground and back
again. Mark all coordinates on the first frame of
the video.
'''
A = []
A.append([270,270]) #Static
A.append([300,300]) #Static
A.append([25,25])   #Static
A.append([400,500]) #Static
A.append([475,130]) #Static
A.append([150,300]) #NonStatic
A.append([156,581]) #NonStatic
A.append([90,553])  #NonStatic
A.append([137,200]) #NonStatic
A.append([150,150]) #Nonstatic


for i in range (0,np.size(A)/2):
    firstframe[A[i][0],A[i][1]] = 255
cv2.imshow("Show first image", firstframe)
#cv2.waitKey()
# When everything done, release the capture


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
plt.scatter(Color_values[0][:][0],Color_values[0][:][1],Color_values[0][:][2])
#plt.scatter(2,2,2)
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_zlim(0,255)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()


cap.release()
cv2.destroyAllWindows()