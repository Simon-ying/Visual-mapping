# load and transform images into array
import cv2 as cv
import numpy as np
from queue import Queue
from matplotlib import pyplot as plt
import open3d as o3d
import os

img_file_path = "./data/camera"
img_files = os.listdir(img_file_path)
orb = cv.ORB_create()

test_img_path = img_file_path + "/" + img_files[0]
test_img = cv.imread(test_img_path)
test_img2 = cv.imread(img_file_path + "/" + img_files[1])

test_img_gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
test_img2_gray = cv.cvtColor(test_img2, cv.COLOR_BGR2GRAY)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

p0 = cv.goodFeaturesToTrack(test_img_gray, mask = None, **feature_params)
kp = orb.detect(test_img_gray)
p01 = np.array([[i.pt[0], i.pt[1]] for i in kp], np.float32)
p01 = p01.reshape((p01.shape[0], 1, p01.shape[1]))
# print(p0.shape, p01.shape, type(p0), type(p01))
# print(p0)
# print(p01)
# print(p0)
mask = np.zeros_like(test_img)
color = np.random.randint(0, 255, (500, 3))
p1, st, err = cv.calcOpticalFlowPyrLK(test_img_gray, test_img2_gray, p01, None, **lk_params)
if p1 is not None:
    good_new = p1[st==1]
    good_old = p01[st==1]
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    frame = cv.circle(test_img2, (int(a), int(b)), 5, color[i].tolist(), -1)
img = cv.add(frame, mask)
cv.imshow('frame', img)
k = cv.waitKey(0)

# kp = orb.detect(test_img)

# kp, des = orb.compute(test_img, kp)
# test_img_kp = cv.drawKeypoints(test_img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(test_img_kp)
# print(test_img2.shape)
# plt.show()