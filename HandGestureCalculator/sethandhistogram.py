import cv2
import numpy as np
import pickle

#imported the libraries
def build_squares(img):
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.vstack((imgCrop, img[y:y+h, x:x+w]))
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
			x+=w+d
		x = 420
		y+=h+d
	return imgCrop

def get_hand_hist():
	cam = cv2.VideoCapture(0)#capturing the video
	x, y, w, h = 300, 100, 300, 300
	flagPressedC, flagPressedS = False, False
	while True:
		img = cam.read()[1]#reading the video will return a boolean value and a frame
		img = cv2.flip(img,1)#flipping the image since we want the mirror image to be processed
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #converting a bgr image to hsv since we rgb components of an image are correlated with the luminance
		keypress = cv2.waitKey(1)#it will wait for 1 milliseconds for any keyboard event
		if keypress == ord('c'):	#ord returns the ascii value	
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
			flagPressedC = True
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])#no of pixels on y axis and pixel values on x axis
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		elif keypress == ord('s'):
			flagPressedS = True	
			break
		if flagPressedC:	
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)#to identifya region of interest
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			thresh = cv2.merge((thresh,thresh,thresh))
			res = cv2.bitwise_and(img,thresh)
			cv2.imshow("res", res)
			cv2.imshow("Thresh", thresh)
		if not flagPressedS:
			imgCrop = build_squares(img)
		#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Set hand histogram", img)
	cam.release()
	cv2.destroyAllWindows()
	with open("hist", "wb") as f:
		pickle.dump(hist, f)


get_hand_hist()