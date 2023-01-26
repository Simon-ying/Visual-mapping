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
kp = orb.detect(test_img)
kp, des = orb.compute(test_img, kp)
test_img_kp = cv.drawKeypoints(test_img, kp, None, color=(0,255,0), flags=0)
plt.imshow(test_img_kp)
plt.show()