import cv2
import numpy as np 
from bs4 import BeautifulSoup
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("imagefolder", help="Input image folder to convert")
parser.add_argument("xml", help="Input xml file to convert")
parser.add_argument("--out", "-o", help="Name of output folder", default="xmlpicks")
# parser.add_argument("--tag", "-t", help="Tag string to add before group number", default="")
args = parser.parse_args()

      # <SpotsInFrame frame="1">

with open(args.xml, 'r') as f:
    data = f.read()

Bs_data = BeautifulSoup(data, "xml")
b_unique = Bs_data.find_all('SpotsInFrame')
 

def integize(stri):
	a=stri[0]
	if a=='F':
		a=0
	elif a=='N':
		a=500
	elif a=='S':
		a=1000
	c=stri.replace('.jpg', '').replace('.xml', '')
	# print(c)
	try:
		b=int(c[8:])
	except:
		a=10000
		b=10000
	return a+b

a = os.listdir(args.imagefolder)
a.sort(key=integize)

out_dir=os.path.join(args.imagefolder, args.out)
if not os.path.exists(out_dir):
	os.mkdir(out_dir)

for i, filename in enumerate(a):
	if os.path.splitext(filename)[1]=='.jpg':
		img = cv2.imread(os.path.sep.join([args.imagefolder,filename]), cv2.COLOR_BGR2RGB)
		children = b_unique[i].findChildren("Spot" , recursive=False)
		for child in children:
			cv2.circle(img, (int(float(child['POSITION_X'])), int(float(child['POSITION_Y']))), int(float(child['RADIUS'])), (0, 0, 255), 2)
		# cv2.imshow(filename, img)
		if not cv2.imwrite(os.path.sep.join([out_dir, filename]), img):
			raise Exception("Could not write image")
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# print(os.path.sep.join([args.imagefolder, args.out, filename]))