import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib.ticker import LinearLocator
from scipy.ndimage import gaussian_filter
from matplotlib.figure import figaspect
import pandas as pd
import vtk
import surf2stl
import stl
from stl import mesh
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hdf5file", help="Input hdf5 file to convert", default="C:/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/cropped_cuboc_original_3D_5nM imager_1nmsample_60percent_54deg_50ms_15K_day_two_take_one_locs_render_filter_filter_picked.hdf5")
parser.add_argument("--outputfolder", help="Output folder for generated images", default="C:/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/render/retry/")
parser.add_argument("--grapht", help="Graph type options: matplotlib, myavi", default="matplotlib")
args = parser.parse_args()


# hdf = np.array(h5py.File('C:/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/cropped_cuboc_original_3D_5nM imager_1nmsample_60percent_54deg_50ms_15K_day_two_take_one_locs_render_filter_filter_picked.hdf5')['locs'])

hdf=np.array(h5py.File(args.hdf5file)['locs'])



def calc_kde(data):
    return kde(data.T)


# for i in range(np.max(hdf['group'])+1):

# for i in range(1):
    #     ind = np.argmax(hdf['group']>i)
        # w, h = figaspect(1)
        # fig, ax = plt.subplots(figsize=(w,h))
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # data = np.histogram2d(hdf['y'][prevind:ind], hdf['z'][prevind:ind], bins=250)[0]
        # data = np.histogramdd([hdf['x'][prevind:ind], hdf['y'][prevind:ind], hdf['z'][prevind:ind]], bins=50)[0]

    # prevind=0

prevind=0
for i in range(np.max(hdf['group'])):
    ind = np.argmax(hdf['group']>i)
    # prevind=np.argmax(hdf['group']>15)
    # prevind=0
    # ind = np.argmax(hdf['group']>16)
    mu, sigma = 0, 0.1 
    x = hdf['x'][prevind:ind]
    y = hdf['y'][prevind:ind]
    z = hdf['z'][prevind:ind]


    if (args.grapht).lower()=="matplotlib":
        x=117*x
        y=117*y

        xyz = np.vstack([x,y,z])
        density = stats.gaussian_kde(xyz)(xyz) 


        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]

        rem_index=int(len(density)*0.5)
        x=x[rem_index:]
        y=y[rem_index:]
        z=z[rem_index:]
        density=density[rem_index:]



        fig = plt.figure(figsize=(1000/120, 750/120), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        surf=ax.scatter(x, y, z, c=density, s=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.axis('off')
        # plt.savefig("C:/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/render/A-group-1.jpg", bbox_inches='tight',pad_inches = 0)
        # plt.show()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(args.outputfolder,"{}-group-iso.jpg".format(i)))
        plt.clf()
        plt.figure().clear()
        plt.cla()
        fig = plt.figure(figsize=(1000/120, 750/120), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=density, s=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.view_init(vertical_axis='z')
        ax.view_init(0,90)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        plt.savefig(os.path.join(args.outputfolder,"{}-group-XZ.jpg".format(i)))
        plt.clf()
        plt.figure().clear()
        plt.cla()
        fig = plt.figure(figsize=(1000/120, 750/120), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=density, s=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(0,180)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(args.outputfolder,"{}-group-YZ.jpg".format(i)))
        plt.clf()
        plt.figure().clear()
        plt.cla()
        fig = plt.figure(figsize=(1000/120, 750/120), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=density, s=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(90,90)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(args.outputfolder,"{}-group-XY.jpg".format(i)))
        plt.clf()
        plt.figure().clear()
        plt.cla()
    

    elif (args.grapht).lower()=='myavi':
        # x=np.interp(x, (x.min(), x.max()), (z.min(), z.max()))
        # y=np.interp(y, (y.min(), y.max()), (z.min(), z.max()))
        z=np.interp(z, (z.min(), z.max()), (x.min(), x.max()))
        y=np.interp(y, (y.min(), y.max()), (x.min(), x.max()))
        xyz = np.vstack([x,y,z])
        # print(xyz)
        kde = stats.gaussian_kde(xyz)
        density = stats.gaussian_kde(xyz)(xyz) 

        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]
        rem_index=int(len(density)*0.5)
        x=x[rem_index:]
        y=y[rem_index:]
        z=z[rem_index:]
        density=density[rem_index:]

        # Evaluate kde on a grid
        xmin, ymin, zmin = x.min(), y.min(), z.min()
        xmax, ymax, zmax = x.max(), y.max(), z.max()
        xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
        density = kde(coords).reshape(xi.shape)


        # Plot scatter with mayavi
        figure = mlab.figure('DensityPlot')

        grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
        min = density.min()
        max=density.max()
        a=mlab.pipeline.volume(grid, vmin=min, vmax=min + .9*(max-min))


        mlab.axes()
        mlab.show()

    else:
        break

    prevind=ind

    

# good picks 6, 7, 15







##########################################################################

# model=np.vstack((x,y,z))    
# model=np.transpose(model)              

# model -= model.mean(axis=0)         
# rad = np.linalg.norm(model, axis=1)    
# zen = np.arccos(model[:,-1] / rad)      
# azi = np.arctan2(model[:,1], model[:,0])
# triang = mtri.Triangulation(zen, azi)  

# # triang=triangulation(x,y,z)
# data = np.zeros(len(triang.triangles), dtype=mesh.Mesh.dtype)
# mobius_mesh = mesh.Mesh(data, remove_empty_areas=False)
# mobius_mesh.x[:] = x[triang.triangles]
# mobius_mesh.y[:] = y[triang.triangles]
# mobius_mesh.z[:] = z[triang.triangles]
# mobius_mesh.save('mysurface4.stl')

##########################################################################

# xyz = np.vstack([x,y,z])
# kde = stats.gaussian_kde(xyz)
# density = kde(xyz)

# # Plot scatter with mayavi
# figure = mlab.figure('DensityPlot')
# pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.02)
# mlab.axes()
# mlab.show()

##########################################################################





##########################################################################

# xyz = np.vstack([x,y,z])
# # print(xyz)
# kde = stats.gaussian_kde(xyz)
# density = stats.gaussian_kde(xyz)(xyz) 

# idx = density.argsort()
# x, y, z, density = x[idx], y[idx], z[idx], density[idx]
# rem_index=int(len(density)*0.5)
# x=x[rem_index:]
# y=y[rem_index:]
# z=z[rem_index:]
# density=density[rem_index:]

# # Evaluate kde on a grid
# xmin, ymin, zmin = x.min(), y.min(), z.min()
# xmax, ymax, zmax = x.max(), y.max(), z.max()
# xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
# coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
# density = kde(coords).reshape(xi.shape)


# # Plot scatter with mayavi
# figure = mlab.figure('DensityPlot')

# grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
# min = density.min()
# max=density.max()
# a=mlab.pipeline.volume(grid, vmin=min, vmax=min + .9*(max-min))


# mlab.axes()
# mlab.show()
