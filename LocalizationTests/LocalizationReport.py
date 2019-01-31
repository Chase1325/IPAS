import csv as csv
import matplotlib as m
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.style
from itertools import product, combinations
import numpy as np
import math as math
import pandas as pd
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.backends.backend_pdf import PdfPages as pdf

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#def create_ellipse(center, lengths, angle = 0)
def create_ellipse(center, lengths):
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    return ell

dataTable = []
meanTable = []
varTable = []
stdTable = []

page = pdf('C:/Users/clstl/Desktop/LocalizationTests/LocalizationReport.pdf')
xy_range = [2,5,8]
pos2mm = 12*25.4
z_range = [850,1435,2005]

locTable = []
for k in z_range:
    for j in xy_range:
        for i in xy_range:
            locTable.append([i*pos2mm,j*pos2mm,k])

#DO THE MATH
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



#GENERATE THE FIGURES

tolerance = 100
scale=250
scale2=250

for i in range(27):
    fig = plt.figure(i, figsize=(8,5))

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
    ax.set_xlim3d(trueX-scale, trueX+scale)
    ax.set_ylim3d(trueY-scale, trueY+scale)
    ax.set_zlim3d(trueZ-scale, trueZ+scale)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

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
        if((trueX-scale<dataTable[i]['X pos'][k]<trueX+scale) and
           (trueY-scale<dataTable[i]['Y pos'][k]<trueY+scale) and
           (trueZ-scale<dataTable[i]['Z pos'][k]<trueZ+scale)):
           ax.scatter(dataTable[i]['X pos'][k],dataTable[i]['Y pos'][k],dataTable[i]['Z pos'][k],
                   color = 'k', alpha = 0.25)

    ax = fig.add_subplot(1,2,2)
    ax.set_xlim(trueX-scale2,trueX+scale2)
    ax.set_ylim(trueY-scale2,trueY+scale2)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    #Estimated Circle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sX*np.cos(u)*np.sin(v)+estX
    y = sY*np.sin(u)*np.sin(v)+estY
    ax.plot(x, y, color='r',alpha=0.5)


    #Actual Circle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = tolerance*np.cos(u)*np.sin(v)+trueX
    y = tolerance*np.sin(u)*np.sin(v)+trueY
    ax.plot(x, y, color='b',alpha=0.5)

    #Draw Measured Data Points
    for k in range(250):
        ax.scatter(dataTable[i]['X pos'][k],dataTable[i]['Y pos'][k],
                   color = 'k', alpha = 0.25)

    ax.arrow(trueX,trueY,estX-trueX,
             estY-trueY, color='g', width=1.5)

    fig.suptitle(f"Measurements at X={int(trueX)}mm, Y={int(trueY)}mm, Z={int(trueZ)}mm",
                 fontsize=14)
    #ax.grid()
    fig.tight_layout(pad=3.0)
    fig.savefig(page, format='pdf')

#Generate Probabilities results
probTable = []
rangeVal = []

for j in range(27):
    tempTable = []
    for i in range(350):
        estCircle = create_ellipse([meanTable[j][0],meanTable[j][1]],
                                   [stdTable[j][0],stdTable[j][1]])
        trueCircle = create_ellipse([locTable[j][0],locTable[j][1]],
                                    [i+1,i+1])
        intersect = estCircle.intersection(trueCircle)
        area = (intersect.area)/(estCircle.area)
        prob = 100*area
        tempTable.extend([prob])
    probTable.append(tempTable)
print(probTable)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.suptitle('Likelihood of Location: Z = 850mm')
ax.set_xlabel(r'$Tolerance: \sigma_{xy}$')
ax.set_ylabel('Probability of Intersection')
rangeVal = np.linspace(1, 350, 350)
for i in range(0,10):
    ax.plot(rangeVal, probTable[i])
ax.legend(['x:609.6,y:609.6', 'x:1524,y:609.6', 'x:2438.4,y:609.6',
           'x:609.6,y:1524', 'x:1524,y:1524', 'x:2438.4,y:1524',
           'x:609.6,y:2438.4', 'x:1524,y:2438.4', 'x:2438.4,y:2438.4'])
fig.savefig(page, format='pdf')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.suptitle('Likelihood of Location: Z = 1435mm')
ax.set_xlabel(r'$Tolerance: \sigma_{xy}$')
ax.set_ylabel('Probability of Intersection')
rangeVal = np.linspace(1, 350, 350)
for i in range(10,19):
    ax.plot(rangeVal, probTable[i])
ax.legend(['x:609.6,y:609.6', 'x:1524,y:609.6', 'x:2438.4,y:609.6',
           'x:609.6,y:1524', 'x:1524,y:1524', 'x:2438.4,y:1524',
           'x:609.6,y:2438.4', 'x:1524,y:2438.4', 'x:2438.4,y:2438.4'])
fig.savefig(page, format='pdf')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.suptitle('Likelihood of Location: Z = 2005mm')
ax.set_xlabel(r'$Tolerance: \sigma_{xy}$')
ax.set_ylabel('Probability of Intersection')
rangeVal = np.linspace(1, 350, 350)
for i in range(19,27):
    ax.plot(rangeVal, probTable[i])
ax.legend(['x:609.6,y:609.6', 'x:1524,y:609.6', 'x:2438.4,y:609.6',
           'x:609.6,y:1524', 'x:1524,y:1524', 'x:2438.4,y:1524',
           'x:609.6,y:2438.4', 'x:1524,y:2438.4', 'x:2438.4,y:2438.4'])
fig.savefig(page, format='pdf')


#POPULATE DATA TABLE RESULTS
celltext = []

for i in range(27):

    estCircle = create_ellipse([meanTable[i][0],meanTable[i][1]],
                               [stdTable[i][0],stdTable[i][1]])

    trueCircle1 = create_ellipse([locTable[i][0],locTable[i][1]],
                                [50,50])
    trueCircle2 = create_ellipse([locTable[i][0],locTable[i][1]],
                                [150,150])
    trueCircle3 = create_ellipse([locTable[i][0],locTable[i][1]],
                                [250,250])
    trueCircle4 = create_ellipse([locTable[i][0],locTable[i][1]],
                                [350,350])

    intersect1 = estCircle.intersection(trueCircle1)
    intersect2 = estCircle.intersection(trueCircle2)
    intersect3 = estCircle.intersection(trueCircle3)
    intersect4 = estCircle.intersection(trueCircle4)

    area1 = (intersect1.area)/(estCircle.area)
    area2 = (intersect2.area)/(estCircle.area)
    area3 = (intersect3.area)/(estCircle.area)
    area4 = (intersect4.area)/(estCircle.area)

    e_x = abs(round(locTable[i][0]-meanTable[i][0], 2))
    e_y = abs(round(locTable[i][1]-meanTable[i][1], 2))
    e_z = abs(round(locTable[i][2]-meanTable[i][2], 2))
    celltext.append([round(locTable[i][0], 2),round(locTable[i][1], 2),
                     round(locTable[i][2], 2),
                     round(meanTable[i][0], 2), round(meanTable[i][1], 2),
                     round(meanTable[i][2], 2),
                     e_x, e_y, e_z,
                     round(math.sqrt(math.pow(e_x,2)+math.pow(e_y,2)),2),
                     round(math.sqrt(math.pow(e_x,2)+math.pow(e_y,2)+math.pow(e_z,2)),2),
                     round(stdTable[i][0], 2),round(stdTable[i][1], 2),
                                      round(stdTable[i][2], 2),
                                      round(area1*100,2),
                                      round(area2*100,2),
                                      round(area3*100,2),
                                      round(area4*100,2)])

columns = ('X [mm]', 'Y [mm]', 'Z [mm]', r'$x_{est} [mm]$', r'$y_{est} [mm]$',
           r'$z_{est} [mm]$', r'$e_x [mm]$',
           r'$e_y [mm]$', r'$e_z [mm]$', r'$D_{xy} [mm]$', r'$D_{xyz} [mm]$',
           r'$\sigma_x [mm]$', r'$\sigma_y [mm]$', r'$\sigma_z [mm]$',
           r'$P[Tol=50]$', r'$P[Tol=150]$', r'$P[Tol=250]$', r'$P[Tol=350]$')

#cell_text = ('X', 'Y', 'Z', "r'$\bar{x}$")
hcell,wcell=0.35, 1
hpad,wpad= 1, 0
nrows,ncols = len(celltext)+1, len(columns)
fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
ax = fig.add_subplot(111)
ax.axis('off')
#
table = ax.table(cellText=celltext,colLabels=columns,loc='center')

plt.savefig('C:/Users/clstl/Desktop/LocalizationTests/LocalizationTable.png')

fig.savefig(page, format='pdf')


#READ IN THE RANGEFINDER DATA
rangeRawTable = []
rangeMeanTable = []
rangeVarTable = []
rangeStdTable = []
rangeErrorTable = []
avgVar = 0
avgStd = 0
avgError = 0

trueRange = [58.6, 87.8, 116.6, 145.5, 187.7, 203.7]

rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-0-586.csv"))
rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-0-878.csv"))
rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-1-166.csv"))
rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-1-455.csv"))
rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-1-877.csv"))
rangeRawTable.append(pd.read_csv(f"C:/Users/clstl/Desktop/LocalizationTests/localizationTest-2-037.csv"))

for i in range(6):
    rangeMeanTable.append(rangeRawTable[i]['Z pos'].mean())
    rangeVarTable.append(rangeRawTable[i]['Z pos'].var())
    rangeStdTable.append(math.sqrt(rangeVarTable[i]))
    rangeErrorTable.append(abs(rangeMeanTable[i]-trueRange[i]))

avgVar = sum(rangeVarTable)/len(rangeVarTable)
avgStd = sum(rangeStdTable)/len(rangeStdTable)
avgError = sum(rangeErrorTable)/len(rangeErrorTable)

#POPULATE DATA TABLE RESULTS
celltext = []

for i in range(6):
    celltext.append([trueRange[i],round(rangeMeanTable[i], 2),
                     round(rangeErrorTable[i], 2), round(rangeVarTable[i],2),
                     round(rangeStdTable[i],2)
                     ])

columns = ('Z [cm]', r'$z_{est} [cm]$', r'$e_z [cm]$',
           r'$\sigma_{z}^2 [cm^2]$', r'$\sigma_z [cm]$')
#cell_text = ('X', 'Y', 'Z', "r'$\bar{x}$")
hcell,wcell=0.35, 1
hpad,wpad= 1, 0
nrows,ncols = len(celltext)+1, len(columns)
fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
ax = fig.add_subplot(111)
ax.axis('off')
#
table = ax.table(cellText=celltext,colLabels=columns,loc='center')

plt.savefig('C:/Users/clstl/Desktop/LocalizationTests/RangeTable.png')

fig.savefig(page, format='pdf')


page.close()
#plt.show()
