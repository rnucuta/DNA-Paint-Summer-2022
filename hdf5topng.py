import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.ndimage import gaussian_filter
from matplotlib.figure import figaspect

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hdf5file", help="Input hdf5 file to convert", default="/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/test2/3 rep for Jonathan/Cropped1_Daria_3redU(3)_2nMImager_10nm_53angle_90k_secondday_Daria_3redU(3)_2nMImager_10nm_53angle_90k_secondday-1_locs_render_picked.hdf5")
parser.add_argument("--outputfolder", help="Output folder for generated images", default="/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/test2/")
parser.add_argument("--filenameprefix", help="File name prefix to append pick number too for output", default="A")
args = parser.parse_args()


#hdf = np.array(h5py.File('/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/Individually imaged NSF For Jonathan and Ritvik/20201213_Bima_S-20nm grid 1nM_R1-5nM/Hand Picked/Cropped_Bima_S-20nm grid 1nM_R1-5nM_TIRF_15Kframes_50ms_70mW_locs_render_picked.hdf5')['locs'])

# hdf = np.array(h5py.File('/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/For Ritvik and Raymond for training data to pick/N/Cropped_Bima_N-20nm grid 1nM_R1-5nM_TIRF_15Kframes_50ms_70mW_locs_render_picked.hdf5')['locs'])

# hdf = np.array(h5py.File('/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/cropped_cuboc_original_3D_5nM imager_1nmsample_60percent_54deg_50ms_15K_day_two_take_one_locs_render_filter_filter_picked.hdf5')['locs'])

# hdf = np.array(h5py.File("/mnt/c/Users/raymo/OneDrive/Documents/DNA Paint/test2/3 rep for Jonathan/Cropped1_Daria_3redU(3)_2nMImager_10nm_53angle_90k_secondday_Daria_3redU(3)_2nMImager_10nm_53angle_90k_secondday-1_locs_render_picked.hdf5")['locs'])

hdf = np.array(h5py.File(args.hdf5file)['locs'])

#convert everyhting to 1:1
#generate 3d graph
#revamp training data

out_dir=os.path.join(args.outputfolder)
if not os.path.exists(out_dir):
	os.mkdir(out_dir)


prevind=0
for i in range(np.max(hdf['group'])):
	ind = np.argmax(hdf['group']>i)
	w, h = figaspect(1)
	fig, ax = plt.subplots(figsize=(w,h))
	# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	data = np.histogram2d(hdf['x'][prevind:ind], hdf['y'][prevind:ind], bins=250)[0]
	# data = np.histogramdd([hdf['x'][prevind:ind], hdf['y'][prevind:ind], hdf['z'][prevind:ind]], bins=50)[0]

	data = gaussian_filter(data, sigma=6)
	# print(data.shape)
	plt.pcolormesh(data.T, cmap='inferno', shading='gouraud')
	# surf = ax.plot_surface(data[0], data[1], data[2], cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.axis('off')
	plt.savefig(os.path.join(args.outputfolder, "{}-group-{}.jpg".format(args.filenameprefix, i), bbox_inches='tight',pad_inches = 0)
	plt.clf()
	prevind=ind

#c=hdf['photons'][:ind]
 