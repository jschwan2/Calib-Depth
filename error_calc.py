# importing mplot3d toolkits
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

all_plane_info = np.load('all_plane_info.npz')
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
for coord in all_coord_info:
	left_x = coord[0]
	left_y = coord[1]
	disparity = coord[2]
	X = coord[3]
	Y = coord[4]
	Z = coord[5]
	if int(Z) < 3:
		xs.append(X)
		ys.append(Y)
		zs.append(Z)
		print(X, Y, Z)

print(len(xs))


fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(xs, zs, ys)


# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# Manual solution
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
print("errors: \n", errors)
print("residual:", residual)

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
print(X,Z,Y)
ax.plot_wireframe(X,Z,Y, color='k')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
plt.show()
