import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
img = cv.imread("./dataset/set_1/original.jpg", cv.IMREAD_UNCHANGED)
# res = cv.resize(img, None, fx=0.02, fy=0.02, interpolation=cv.INTER_LINEAR)
# res = cv.blur(img, (100, 100))
res = cv.blur(img, (100, 100))
# cv.imwrite("./dataset/set_1/copy_2.jpg", res)
print(np.float32([[1, 0, 10], [2, 9, 10]]))
plt.subplot(1, 2, 1)
plt.title("unsized")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.subplot(1, 2, 2)

plt.title("sized")
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
print(res.shape[1:2])

