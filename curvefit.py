import cv2
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, imsave
import numpy as np
import pydicom as dcm
from skimage import morphology

#img=imread('rice.png')
#img=cv2.imread('rice.png',0)
ds=dcm.dcmread('D:\\Bigdata\\ha\\curve\\breast3.dicom')
img=ds.pixel_array

hist,bins=np.histogram(img,bins=256)
X=bins[0:-1]

p=np.polyfit(X,hist,6)

Y=np.polyval(p,X)

dY=np.diff(Y)
d2Y=np.diff(dY)

dY_abs=np.sort(np.abs(dY))
ind=np.argsort(np.abs(dY))

d2Y_abs=np.diff(dY_abs)
voluem=np.min(d2Y_abs)
i=np.argmin(d2Y_abs)

thresh=ind[i]

thresh_img=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)[1]
thresh_img = thresh_img.astype(np.uint8)
(_, cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented = max(cnts, key=cv2.contourArea)

(x,y,w,h) = cv2.boundingRect(segmented)
crop_img=img[y:y+h,x:x+w]

#cv2.rectangle(thresh_img,(x,y),(x+w, y+h), (255,255,255), 2)
#plt.plot(X,hist,'r-',X,Y,'b-')

#plt.imshow(crop_img)
plt.imshow(thresh_img)
plt.show()