import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import stl
import matplotlib.tri as mtri
from stl import mesh

from scipy import stats
from skimage import measure
from skimage.draw import ellipsoid

# https://github.com/WoLpH/numpy-stl/issues/89



# Generate a level set about zero of two identical ellipsoids in 3D
# ellip_base = ellipsoid(6, 10, 16, levelset=True)
# ellip_double = np.concatenate((ellip_base[:-1, ...],
#                                ellip_base[2:, ...]), axis=0)

# # Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)

# # Display resulting triangular mesh using Matplotlib. This can also be done
# # with mayavi (see skimage.measure.marching_cubes docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

# ax.set_xlabel("x-axis: a = 6 per ellipsoid")
# ax.set_ylabel("y-axis: b = 10")
# ax.set_zlabel("z-axis: c = 16")

# ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0, 20)  # b = 10
# ax.set_zlim(0, 32)  # c = 16

# plt.tight_layout()
# plt.show()


hdf = np.array(h5py.File('C:/Users/raymo/OneDrive/Documents/DNA Paint/3d picks/cropped_cuboc_original_3D_5nM imager_1nmsample_60percent_54deg_50ms_15K_day_two_take_one_locs_render_filter_filter_picked.hdf5')['locs'])


prevind=0
for i in range(np.max(hdf['group'])):
    ind = np.argmax(hdf['group']>i)

    x = hdf['x'][prevind:ind]
    y = hdf['y'][prevind:ind]
    z = hdf['z'][prevind:ind]

    x=np.interp(x, (x.min(), x.max()), (z.min(), z.max()))
    y=np.interp(y, (y.min(), y.max()), (z.min(), z.max()))

    xyz = np.vstack([x,y,z])
    density = stats.gaussian_kde(xyz)(xyz) 
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]
    rem_index=int(len(density)*0.35)
    x=x[rem_index:]
    y=y[rem_index:]
    z=z[rem_index:]
    density=density[rem_index:]

    model=np.vstack((x,y,z))    
    model=np.transpose(model)              

    model -= model.mean(axis=0)         
    rad = np.linalg.norm(model, axis=1)    
    zen = np.arccos(model[:,-1] / rad)      
    azi = np.arctan2(model[:,1], model[:,0])
    triang = mtri.Triangulation(zen, azi)  
    # print(triang.triangles[0])
    # print()

    # triang=triangulation(x,y,z)
    data = np.zeros(len(triang.triangles), dtype=mesh.Mesh.dtype)
    mobius_mesh = mesh.Mesh(data, remove_empty_areas=False)
    mobius_mesh.x[:] = x[triang.triangles]
    mobius_mesh.y[:] = y[triang.triangles]
    mobius_mesh.z[:] = z[triang.triangles]
    # print(mobius_mesh.x)
    # print(len(mobius_mesh.x))
    # print(len(triang.triangles))
    # print()
    mobius_mesh.save('mysurface-{}.stl'.format(i))

    prevind=ind