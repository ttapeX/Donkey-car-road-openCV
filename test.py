import cv2 as cv
import numpy as np

img=cv.imread('test7.jpg')
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
lower_white=np.array([0,0,200])
upper_white=np.array([255,55,255])

masked=cv.inRange(hsv,lower_white,upper_white)
res=cv.bitwise_and(img,img,mask=masked)
greyres=cv.cvtColor(res,cv.COLOR_BGR2GRAY)


#blur=cv.GaussianBlur(img,(5,5),0)
#sigma = np.std(blur)
#mean = np.mean(blur)
#lower = int(max(0, (mean - sigma)))
#upper = int(min(255, (mean + sigma)))
#canny=cv.Canny(blur,lower,upper)

#rectangle=cv.rectangle(canny,(0,0),(160,48),(0,0,0),-1)
#mask=cv.bitwise_and(rectangle,canny)

#thresh=cv.threshold(grey,180,255,cv.THRESH_BINARY)[1]
#kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))
#dilate=cv.morphologyEx(thresh,cv.MORPH_DILATE,kernel)
#diff=cv.absdiff(dilate,thresh)
#edges=255-diff


#blur=cv.GaussianBlur(mask,(3,3),0)
#canny=cv.Canny(blur,85,255)



grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #
imgHLS=cv.cvtColor(img,cv.COLOR_BGR2HLS) #
Lchannel=imgHLS[:,:,1]#
maskHLS=cv.inRange(Lchannel,200,255)#
resHLS=cv.bitwise_and(img,img,mask=maskHLS)#

rectangle=cv.rectangle(grey,(0,0),(160,48),(0,0,0),-1)#
mask=cv.bitwise_and(rectangle,grey)#
no=cv.bitwise_not(rectangle)#

v=np.median(no)#
sigma=0.33#
lower=int(max(0,(1.0-sigma)*v))#
upper=int(min(255,(1.0+sigma)*v))#


greyHLS=cv.cvtColor(resHLS,cv.COLOR_BGR2GRAY)#
kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))#
canny=cv.Canny(mask,lower,upper)#
dilate=cv.dilate(canny,kernel, iterations=1)#

mask2=cv.bitwise_and(dilate,greyHLS)#







cv.imshow('result',mask2)

cv.waitKey(0)