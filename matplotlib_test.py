import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('timg.jpg',0) #直接读为灰度图像
img = cv2.imread('C:\\Users\\34323\\Desktop\\python\\t.png',0)

# 直方图均衡化
res = cv2.equalizeHist(img)

# 自适应均衡化图像
clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
cl1 = clahe.apply(img)

plt.subplot(131),plt.imshow(img,'gray')
plt.subplot(132),plt.imshow(res,'gray')
plt.subplot(133),plt.imshow(cl1,'gray')

plt.show()
