# https://www.dataleek.io/archive/interpolating-3d.html

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata as gd


interpolationmethod = 'cubic'
p = 2
extrapolation_interval = 0.1

results = pickle.load(open('results/densenet_opt.p', 'rb'))

NEURONS = [32,64,128,256,1024]

top = []

for item in results:
    coords = item['misc']['vals']
    # print(coords)
    x = NEURONS[coords['Dense'][0]]
    y = coords['Dropout'][0]
    z = item['result']['loss']
    top.append([np.log2(x),y,z])

results = pickle.load(open('results/densenet_opt_512.p', 'rb'))
for item in results:
    coords = item['misc']['vals']
    # print(coords)
    x = 512
    y = coords['Dropout'][0]
    z = item['result']['loss']
    top.append([np.log2(x),y,z])
top.pop(-3)
top.pop(-4)
# x = np.asarray(x)
# y = np.asarray(y)
# z = np.asarray(z)
# print(x.shape)
# print(y.shape)
# print(z.shape)
# top = np.concatenate((x.reshape((x.shape[0],1)), y.reshape((x.shape[0],1))), axis=1)
top = np.array(top)

print(top[np.argmin(top[:,2])])


def nearest_neighbor_interpolation(data, x, y, p=0.5):
    """
    Nearest Neighbor Weighted Interpolation
    http://paulbourke.net/miscellaneous/interpolation/
    http://en.wikipedia.org/wiki/Inverse_distance_weighting

    :param data: numpy.ndarray
        [[float, float, float], ...]
    :param p: float=0.5
        importance of distant samples
    :return: interpolated data
    """
    n = len(data)
    vals = np.zeros((n, 2), dtype=np.float64)
    distance = lambda x1, x2, y1, y2: (x2 - x1)**2 + (y2 - y1)**2
    for i in range(n):
        vals[i, 0] = data[i, 2] / (distance(data[i, 0], x, data[i, 1], y))**p
        vals[i, 1] = 1          / (distance(data[i, 0], x, data[i, 1], y))**p
    z = np.sum(vals[:, 0]) / np.sum(vals[:, 1])
    return z

def extrapolation(data, extrapolation_spots, method='nearest'):
    if method == 'kriging':
        xx, yy, zz, ss = kriging(data, extrapolation_spots)

        new_points = np.zeros((len(yy) * len(zz), 3))
        count = 0
        for i in range(len(xx)):
            for j in range(len(yy)):
                new_points[count, 0] = xx[i]
                new_points[count, 1] = yy[j]
                new_points[count, 2] = zz[i, j]
                count += 1
        combined = np.concatenate((data, new_points))
        return combined

    if method == 'nearest':
        new_points = np.zeros((len(extrapolation_spots), 3))
        new_points[:, 0] = extrapolation_spots[:, 0]
        new_points[:, 1] = extrapolation_spots[:, 1]
        for i in range(len(extrapolation_spots)):
            new_points[i, 2] = nearest_neighbor_interpolation(data,
                                    extrapolation_spots[i, 0],
                                    extrapolation_spots[i, 1], p=p)
        combined = np.concatenate((data, new_points))
        return combined

def get_plane(xl, xu, yl, yu, i):
    xx = np.arange(xl, xu, i)
    yy = np.arange(yl, yu, i)
    extrapolation_spots = np.zeros((len(xx) * len(yy), 2))
    count = 0
    for i in xx:
        for j in yy:
            extrapolation_spots[count, 0] = i
            extrapolation_spots[count, 1] = j
            count += 1
    return extrapolation_spots

def interpolation(data):
    gridx, gridy = np.mgrid[5:10:50j, 0:1:50j]
    gridz = gd(data[:, :2],data[:, 2], (gridx, gridy),
                method=interpolationmethod)
    return gridx, gridy, gridz



def plot(data, gridx, gridy, gridz, method='rotate', title='nearest', both=False):
    def update(i):
        ax.view_init(azim=i)
        return ax,

    if method == 'rotate':
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        ax.plot_surface(gridx, gridy, gridz*255, alpha=0.8, cmap=cm.terrain, vmin=np.nanmin(gridz*255), vmax=np.nanmax(gridz*255))
        ax.scatter(data[:, 0][np.argmin(data[:, 2])], data[:, 1][np.argmin(data[:, 2])], np.min(data[:, 2])*255, c='red')

        animation.FuncAnimation(fig, update, np.arange(360 * 5), interval=1)
        ax.set_xlabel('Number of hidden neurons (2^x)')
        ax.set_ylabel('Dropout rate')
        ax.set_zlabel('Validation error')
        plt.show()

    elif method== 'snaps':
        fig = plt.figure(figsize=(10, 10))
        angles = [45, 120, 220, 310]

        if both:
            for i in range(1,4):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                ax.plot_wireframe(gridx[0], gridy[0], gridz[0], alpha=0.5)
                ax.plot_wireframe(gridx[1], gridy[1], gridz[1], alpha=0.5)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
                ax.view_init(azim=angles[i])
        else:
            for i in range(1,4):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
                ax.view_init(azim=angles[i])

        plt.savefig('snaps_{}.png'.format(title))

    elif method == 'contour':
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        ax.contourf(gridx, gridy, gridz, zdir='z', offset=np.min(data[:, 2]), cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='x', offset=0, cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='y', offset=0, cmap=cm.coolwarm)
        ax.view_init(azim=45)
        plt.show()




extrapolation_spots = get_plane(0, 10, 0, 1, extrapolation_interval)
# nearest_analysis(extrapolation_spots)

top_extra = extrapolation(top, extrapolation_spots, method='nearest')
gridx_top, gridy_top, gridz_top = interpolation(top_extra)
plot(top, gridx_top, gridy_top, gridz_top, method='rotate',
        title='_top_nearest')



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
plt.show()

# X, Y, Z = axes3d.get_test_data(0.05)
# print(X,Y,Z)