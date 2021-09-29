import pdb
import numpy as np
import cv2
import glob
from tqdm import tqdm
from scipy.linalg import norm
from scipy import sum, average
import pickle
import matplotlib.path as mplPath
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def shortest_distance(x1, y1, z1, a, b, c, d):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    # print("Perpendicular distance is", d/e)
    return d/e

all_plane_info = np.load('board_corner_info.npz')
fx = all_plane_info['fx']
baseline = all_plane_info['baseline']
board_position = all_plane_info['board_position']
board_roll = all_plane_info['board_roll']
board_yaw = all_plane_info['board_yaw']
board_pitch = all_plane_info['board_pitch']
bsboard_position = all_plane_info['bsboard_position']
bsboard_roll = all_plane_info['bsboard_roll']
bsboard_yaw = all_plane_info['bsboard_yaw']
bsboard_pitch = all_plane_info['bsboard_pitch']
all_coord_info = all_plane_info['all_coord_info']

xs = []
ys = []
zs = []


coords_to_use = []
for coord in all_coord_info[::1]:
    left_x = coord[0]
    left_y = coord[1]
    disparity = coord[2]
    X = coord[3]
    Y = coord[4]
    Z = coord[5]
    # if Z < 2.00 and Z > 1.90:
    coords_to_use.append([X*100, Y*100, Z*100])

smallest_x = 10000
smallest_y = 10000
smallest_z = 10000
coord_to_sub = [0,0,0]
for coord in coords_to_use:
    if coord[0] < smallest_x:
        smallest_x = coord[0]
        coord_to_sub[0] = coord[0]
    if coord[1] < smallest_y:
        smallest_y = coord[1]
        coord_to_sub[1] = coord[1]
    if coord[2] < smallest_z:
        smallest_z = coord[2]
        coord_to_sub[2] = coord[2]


for coord in coords_to_use:
    xs.append(coord[0]-coord_to_sub[0])
    ys.append(coord[1]-coord_to_sub[1])
    zs.append(coord[2]-coord_to_sub[2])
    # xs.append(coord[0])
    # ys.append(coord[1])
    # zs.append(coord[2])
print(coords_to_use)

points = [xs, ys, zs]
# subtract out the centroid and take the SVD
svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

# Extract the left singular vectors
left = svd[0]
normal = left[:, -1]

print(left)
print(normal)

point  = np.array([0,0,0])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
a = normal[0]
b = normal[1]
c = normal[2]
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(23), range(20))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

all_dists = []
for i in range(len(xs)):
    distance_from_pt = shortest_distance(xs[i], ys[i], zs[i], a, b, c, d)
    all_dists.append(distance_from_pt)

print("AVG:", sum(all_dists)/len(all_dists))
print("MAX:", max(all_dists))

########
# Create the figure
fig = plt.figure()
# Add an axes
ax = fig.add_subplot(111,projection='3d')
# plot the surface
ax.plot_surface(xx, yy, z, alpha=0.8)
# and plot the point 
ax.scatter(xs , ys , zs,  color='green')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

X = np.array(xs)
Y = np.array(ys)
Z = np.array(zs)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

plt.grid()
plt.show()



