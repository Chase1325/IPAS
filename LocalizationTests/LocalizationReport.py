import csv as csv
import matplotlib as m
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from itertools import product, combinations
import numpy as np
import math as math
import pandas as pd
#import plotly.plotly as py

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#with open("C:/Users/clstl/Desktop/LocalizationTests/localizationTest-2-2-1.csv", 'r') as file:
#    data = list(csv.reader(file))
dataTable = []
meanTable = []
varTable = []
stdTable = []

xy_range = [2,5,8]
pos2mm = 12*25.4
z_range = [850,1435,2005]

count = 0
for k in range(1,4):
    for j in xy_range:
        for i in xy_range:
            data = pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-{i}-{j}-{k}.csv")
            dataTable.append(data)


            #Calculate the Means
            mean_x = dataTable[count]['X pos'].mean()
            mean_y = dataTable[count]['Y pos'].mean()
            mean_z = dataTable[count]['Z pos'].mean()

            mean = [mean_x, mean_y, mean_z]
            meanTable.append(mean)

            #Calculate the Variances
            var_x = dataTable[count]['X pos'].var()
            var_y = dataTable[count]['Y pos'].var()
            var_z = dataTable[count]['Z pos'].var()

            var = [var_x, var_y, var_z]
            varTable.append(var)

            std = [math.sqrt(var_x)/2, math.sqrt(var_y)/2, math.sqrt(var_z)/2]
            stdTable.append(std)
            count += 1

#Draw Field as Cube
#r = [-1, 1]
#for s, e in combinations(np.array(list(product(r, r, r))), 2):
#    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        #ax.plot3D(*zip(s, e), color="b")

locTable = []
tolerance = 50

for k in z_range:
    for j in xy_range:
        for i in xy_range:
            locTable.append([i*pos2mm,j*pos2mm,k])

#Draw Estimated Spheres
#for i in range(27):
for i in range(27):
    fig = plt.figure(i)

    ax = fig.add_subplot(1,2,1, projection='3d')

    trueX = locTable[i][0]
    trueY = locTable[i][1]
    trueZ = locTable[i][2]

    estX = meanTable[i][0]
    estY = meanTable[i][1]
    estZ = meanTable[i][2]

    sX = stdTable[i][0]
    sY = stdTable[i][1]
    sZ = stdTable[i][2]

    #ax.set_aspect("equal")
    ax.set_xlim3d(,3050)
    ax.set_ylim3d(0,3050)
    ax.set_zlim3d(0,3050)

    #Estimated Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sX*np.cos(u)*np.sin(v)+estX
    y = sY*np.sin(u)*np.sin(v)+estY
    z = sZ*np.cos(v)+estZ
    ax.plot_surface(x, y, z, color='r',alpha=0.5)

    #Actual Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = tolerance*np.cos(u)*np.sin(v)+trueX
    y = tolerance*np.sin(u)*np.sin(v)+trueY
    z = tolerance*np.cos(v)+trueZ
    ax.plot_surface(x, y, z, color='b',alpha=0.5)

    #Arrow from Actual centre to Estimated Centre
    a = Arrow3D([trueX, estX], [trueY, estY],
                [trueZ, estZ], mutation_scale=20,
            lw=2, arrowstyle="-|>", color="g")
    ax.add_artist(a)

    #Draw Dots for Origins
    #True Center
    ax.scatter(trueX,trueY,trueZ,color='b')
    #Estimate Center
    ax.scatter(estX,estY,estZ,color='r')

    #Draw estimated values Scatter
    for k in range(250):
        ax.scatter(dataTable[i]['X pos'][k],dataTable[i]['Y pos'][k],dataTable[i]['Z pos'][k],
                   color = 'k', alpha = 0.25)

    ax = fig.add_subplot(1,2,2)

    #Estimated Circle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sX*np.cos(u)*np.sin(v)+estX
    y = sY*np.sin(u)*np.sin(v)+estY
    ax.plot(x, y, color='r')

    #Actual Circle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = tolerance*np.cos(u)*np.sin(v)+trueX
    y = tolerance*np.sin(u)*np.sin(v)+trueY
    ax.plot(x, y, color='b')

    #Draw Measured Data Points
    for k in range(250):
        ax.scatter(dataTable[i]['X pos'][k],dataTable[i]['Y pos'][k],
                   color = 'k', alpha = 0.25)

    ax.arrow(trueX,trueY,estX-trueX,
             estY-trueY, color='g')


plt.show()
