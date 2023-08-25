import cv2 as cv
import numpy as np

print(cv.__version__)

img = cv.imread('images/halloween.jpg',cv.IMREAD_COLOR)

print('shape=', img.shape)

w = img.shape[1]
h = img.shape[0]

print(img.dtype)

new_img = 1.15*img
new_img = new_img.astype(np.uint8)
print(new_img.dtype)

# access pixel value where x = 200, y = 300
a = img[300,200] # read value
img[300,200] = 230 # set value

# crop  image x = [100:200], y = [200:300]
roi=img[200:300, 100:200]

#cv.imshow('image', img)
cv.imshow('new_imgae', roi)
cv.waitKey(0)
cv.destroyAllWindows()