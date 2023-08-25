import cv2 as cv
import numpy as np

# Ảnh đề bài
ques = cv.imread('images/out.PNG', cv.IMREAD_COLOR)

# Add ảnh
img = cv.imread('images/anchor.png', cv.IMREAD_COLOR)

# CHuyển về ảnh xám
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Nhị phân ảnh xám
b_img = cv.threshold(gray,150,255,cv.THRESH_BINARY)[1]
w = b_img.shape[1]
h = b_img.shape[0]

# xac danh y1 va y2
mean_y = np.mean(b_img, axis=1)
y0= np.min(np.where(mean_y>0))
y1= np.min(np.where(mean_y>150))
y2= np.max(np.where(mean_y>150))

# xac dinh x1-> x4
mean_x = np.mean(b_img,axis =0)
x1 = np.min(np.where(mean_x>0))
x2 = np.min(np.where((mean_x>0) & (mean_x<120)))
x3 = np.max(np.where((mean_x>0) & (mean_x<120)))
x4 = np.max(np.where(mean_x>0))

# create new image
# a
new_img_a = np.zeros((h,w), dtype=np.uint8)
new_img_a[y1:y2, x1:x2] = 255
new_img_a[y1:y2, x3:x4] = 255
# b 
new_img_b = np.zeros((h,w), dtype=np.uint8)
new_img_b[y0:y2, x1:x2] = 255
new_img_b[y0:y2, x3:x4] = 255
# test
test = np.zeros((h,w), dtype=np.uint8)
test[y1:y2] = 255
test[y1:y2] = 255

print("y1=",y1)
print("y2=",y2)
print("x1=",x1)
print("x2=",x2)
print("x3=",x3)
print("x4=",x4)

# Kernel
kernel = np.ones((5,5),np.uint8)

kernel_cus =  np.array([[1,0,1],
                      [1,1,1],
                      [0,1,0]], dtype=np.uint8)

kernel_sq =  np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]], dtype=np.uint8)

kernel_cr = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]], dtype=np.uint8)

kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1))

# Morphology
out_a = cv.morphologyEx(new_img_a,cv.MORPH_ERODE,kernel_sq,iterations=10)
testafsd = cv.dilate(b_img,kernel,iterations = 10)
out3 = cv.morphologyEx(test,cv.MORPH_ERODE,kernel,iterations=5)

# cv.imshow('image', img)
cv.imshow('de bai', ques)
# cv.imshow('Nhi phan', b_img)
cv.imshow('a', out_a)
cv.imshow('b', new_img_b)
# cv.imshow('test', test)
# cv.imshow('new image', new_img_a)
cv.waitKey(0)
cv.destroyAllWindows()