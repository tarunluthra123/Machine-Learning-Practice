import cv2
import numpy as np
import pandas as pd
# CV2 version 4.0.0.21

img = cv2.imread('Before.png')

glasses = cv2.imread('glasses.png',-1)
mustache = cv2.imread('mustache.png',-1)

eye_cascade = cv2.CascadeClassifier('frontalEyes35x16_old.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_frontal_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')


eyes = eye_cascade.detectMultiScale(img,1.3,5)

# print(eyes)


# glass=cv2.resize(glasses,(h,w))

# for i in range(glass.shape[0]):
#     for j in range(glass.shape[1]):
#         if(glass[i,j,-1]>0):
#         	img[y+i,x+j,:]=glass[i,j,:-1]
# cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)

for eye in eyes:
	x,y,w,h = eye
	glass=cv2.resize(glasses,(h,w))

	for i in range(glass.shape[0]):
	    for j in range(glass.shape[1]):
	        if(glass[i,j,-1]>0):
	        	img[y+i,x+j,:]=glass[i,j,:-1]
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

nose = nose_cascade.detectMultiScale(img,1.3,5)

print('nose = ',len(nose))

for n in nose:
	x,y,w,h = n
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

	# w,h = int(w/2), int(h/2)
	y = y+int(h/2)

	mstc = cv2.resize(mustache,(h,w))

	for i in range(mstc.shape[0]):
		for j in range(mstc.shape[1]):
			if (mstc[i,j,-1]>0):
				img[y+i,x+j,:] = mstc[i,j,:-1]

cv2.imshow("After-",img)
cv2.waitKey(0)
cv2.destroyAllWindows()