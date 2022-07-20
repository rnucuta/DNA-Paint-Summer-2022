import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, measure, color

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

import os
from PIL import Image

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imagedirectory", help="Input image folder to convert")
args = parser.parse_args()

#https://stackoverflow.com/questions/54425093/how-can-i-find-the-center-of-the-pattern-and-the-distribution-of-a-color-around


################   HSV MAP    #######################
# nemo = cv2.imread('./train1/F-group-1.jpg')
# nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

# r, g, b = cv2.split(nemo)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

# pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()


# hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
# h, s, v = cv2.split(hsv_nemo)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()


###################   Finding color values    #################################

light_orange = (40,94,255)
dark_orange = (1,255,180)

# light_orange = (1, 100, 100)
# dark_orange = (18, 255, 255)

# lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
# do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

# plt.subplot(1, 2, 1)
# plt.imshow(hsv_to_rgb(do_square))
# plt.subplot(1, 2, 2)
# plt.imshow(hsv_to_rgb(lo_square))
# plt.show()



#################### Using color values ##########################
# directory = 'train1'
directory=args.imagedirectory
out_dir=os.path.abspath(os.path.join(directory, '..', 'train2'))
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
for filename in tqdm(os.scandir(directory)):
	if os.path.splitext(filename.path)[1]=='.jpg':
		sph = cv2.imread(filename.path)
		sph = cv2.cvtColor(sph, cv2.COLOR_BGR2RGB)

		hsv_sph = cv2.cvtColor(sph, cv2.COLOR_RGB2HSV)

		mask = cv2.inRange(hsv_sph, light_orange, dark_orange)

		result = cv2.bitwise_and(sph, sph, mask=mask)


		red_image = result[:,:,1]
		red_th = filters.threshold_otsu(red_image)


		red_mask = red_image > red_th;
		red_mask.dtype ;

		# io.imshow(red_image)
		# io.show()

		ret,thresh = cv2.threshold(sph,200,255,cv2.THRESH_BINARY)

		kernel = np.ones((2,2),np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY);
		opening = cv2.convertScaleAbs(opening)

		contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		areas = []

		centersX = []
		centersY = []

		for cnt in contours:

		    areas.append(cv2.contourArea(cnt))

		    M = cv2.moments(cnt)
		    centersX.append(int(M["m10"] / M["m00"]))
		    centersY.append(int(M["m01"] / M["m00"]))


		full_areas = np.sum(areas)

		acc_X = 0
		acc_Y = 0

		for i in range(len(areas)):

		    acc_X += centersX[i] * (areas[i]/full_areas) 
		    acc_Y += centersY[i] * (areas[i]/full_areas)

		# print (acc_X, acc_Y) 
		# cv2.circle(sph, (int(acc_X), int(acc_Y)), 5, (255, 0, 0), -1)

		

		(H,W) = sph.shape[:2]

		left=int(max(acc_X-174, 0))
		top=int(min(H,acc_Y+129))
		right=int(min(W, acc_X+129))
		bottom=int(max(0, acc_Y-129))

		cropped_img = sph[bottom:top, left:right]
		cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)

		# plt.imshow(cropped_img)
		# plt.show()

		cv2.imwrite(os.path.join(out_dir,filename.name), cropped_img)