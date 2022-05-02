import cv2
import numpy as np


# img = cv2.imread('C:\\Users\\34323\\Desktop\\python\\t.png', cv2.IMREAD_COLOR)

# 变为灰度图片
img = cv2.imread('C:\\Users\\34323\\Desktop\\python\\t.png', 0)
cv2.imshow("image", img)
cv2.waitKey(0)
