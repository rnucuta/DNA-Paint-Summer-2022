# DNA-Paint-SummerResearch-2022
The following code is either helper code for this [binary classifier](https://colab.research.google.com/drive/1elS4sCJqXg2ybNFYJ3sbeeM9x0Wj5w4v?usp=sharing) or has other utility for the Biodesign Lab.

## `hdf5topng.py`
This script converts all groups (picks) in an hdf5 file to an image. Primary use of this was to generate training/testing data for the binary classifier.

## `train_crop.py`
This script finds the center of a training pick and crops it to 70% of its size while keeping the new center of the image. Intended to clean up noise from training data.

## `imagefiji.py`
This script takes Fiji Trackmate xml data and displays the picks on images for viewing purposes.

## `3d_render.py`
This script is for DNA paint localizations in 3 dimensions. It either exports 4 different views (isometric, xy, xz, yz) or displays a surface for each grouping.

## `stl_render.py`
This script is for DNA paint localizations in 3 dimensions. It renders an stl file for every grouping in an hdf5 file. The stl file exports poorly due to the grainy nature of the data.


