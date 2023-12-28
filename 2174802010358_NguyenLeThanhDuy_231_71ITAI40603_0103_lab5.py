import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:\ThiGiacMayTinh\LAB6/stop.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load the cascade
stop_data = cv2.CascadeClassifier('D:\ThiGiacMayTinh\LAB6/stop_data.xml')

# Detect STOP signs
found = stop_data.detectMultiScale(img_gray, minSize=(20, 20))
amount_found = len(found)

if amount_found != 0:
    for (x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

plt.imshow(img_rgb)
plt.show()


import cv2
from matplotlib import pyplot as nltd
   
# Opening image
img = cv2.imread("D:\ThiGiacMayTinh/homies.jpg")
   
# OpenCV opens images as BRG 
# but we want it as RGB We'll 
# also need a grayscale version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   
# Use minSize because for not 
# bothering with extra-small 
# dots that would look like STOP signs
face_data = cv2.CascadeClassifier('D:\ThiGiacMayTinh/haarcascade_frontalface_default.xml')
   
found = face_data.detectMultiScale(img_gray, minSize =(20, 20))
   
# Don't do anything if there's 
# no sign
amount_found = len(found)
   
if amount_found != 0:
       
    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
           
        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(img_rgb, (x, y), 
                      (x + width, y + height), 
                      (0, 255, 0), 2)
           
# Creates the environment of 
# the picture and shows it
nltd.imshow(img_rgb)
nltd.show()