from slidingWindow import pyramid
from slidingWindow import sliding_window
import time
import cv2 as cv

# load the image and define the window width and height
image = cv.imread('img/road-signs.jpg', cv.IMREAD_COLOR)
(winW, winH) = (128, 128)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv.imshow("Window", clone)
		cv.waitKey(1)
		time.sleep(0.025)