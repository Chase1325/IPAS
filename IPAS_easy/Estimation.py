
from numpy.linalg import inv
from scipy import misc

import numpy as np
import matplotlib.pyplot as plt
import math

#----------------------------------------------------------------------------------------#
# Training data

x1_data = np.array([-8.0, -8.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -7.9, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])

y_data = np.array([0.96, 0.962, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

sigma_n = 0.1

x1_min = min(x1_data) - 2.0
x1_max = max(x1_data) + 2.0

x2_min = min(x2_data) - 2.0
x2_max = max(x2_data) + 2.0

matplotlib_marker = []
for i in y_data:
	if i > 0: matplotlib_marker.append('o')
	if i <= 0: matplotlib_marker.append('x')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title('Training Data')

plt.grid(True)

for i in range(len(x1_data)):
	plt.scatter(x1_data[i], x2_data[i], marker=matplotlib_marker[i])

plt.savefig('gaussian_processes_2d_training_data.png', bbox_inches='tight')
#plt.show()
plt.close()

#----------------------------------------------------------------------------------------#
# Find Gaussian processes (3) hyperparameters using gradient descent

def log_likelihood_function(gph_l1, gph_l2, gph_sigma_f):
	dim_x_data = len(x1_data)
	k = np.zeros((dim_x_data,dim_x_data))
	for i in range(dim_x_data):
		for j in range(i+1):
			dx1 = x1_data[i] - x1_data[j]
			dx2 = x2_data[i] - x2_data[j]
			coef = - 0.5 * ( dx1**2 / gph_l1**2 + dx2**2 / gph_l2**2 )
			k[i,j] = gph_sigma_f * gph_sigma_f * math.exp( coef )
			k[j,i] = k[i,j]
			if i == j:
				k[i,j] = k[i,j] + sigma_n * sigma_n
	k_inv = inv(k)
	m1 = np.dot(k_inv,y_data)
	part1 = np.dot(y_data.T,m1)
	part2 = math.log(np.linalg.det(k))
	return -0.5 * part1 - 0.5 * part2 - dim_x_data / 2.0 * math.log(2*math.pi)

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return misc.derivative(wraps, point[var], dx = 1e-6)

alpha = 0.1 # learning rate
nb_max_iter = 100 # Nb max of iterations
eps = 0.0001 # stop condition

gph_l1 = 2.5 # start point
gph_l2 = 2.5 # start point
gph_sigma_f = 3.0 # start point

ll_value = log_likelihood_function(gph_l1, gph_l2, gph_sigma_f)

cond = eps + 10.0 # start with cond greater than eps (assumption)
nb_iter = 0
tmp_ll_value = ll_value
while cond > eps and nb_iter < nb_max_iter:

	tmp_gph_l1 = gph_l1 + \
	alpha * partial_derivative(log_likelihood_function, 0, [gph_l1, gph_l2, gph_sigma_f])

	tmp_gph_l2 = gph_l2 + \
	alpha * partial_derivative(log_likelihood_function, 1, [gph_l1, gph_l2, gph_sigma_f])

	tmp_gph_sigma_f = gph_sigma_f + \
	alpha * partial_derivative(log_likelihood_function, 2, [gph_l1, gph_l2, gph_sigma_f])

	gph_l1 = tmp_gph_l1
	gph_l2 = tmp_gph_l2
	gph_sigma_f = tmp_gph_sigma_f

	ll_value = log_likelihood_function(gph_l1, gph_l2, gph_sigma_f)

	nb_iter = nb_iter + 1

	cond = abs( tmp_ll_value - ll_value )

	tmp_ll_value = ll_value

	print('gph_l1, gph_l2, gph_sigma_f, nb_iter, cond', \
        gph_l1, gph_l2, gph_sigma_f, nb_iter, cond)

print("Hyperparameters found using gradient descent: ")
print('gph_l1, gph_l2, gph_sigma_f', gph_l1, gph_l2, gph_sigma_f)

#----------------------------------------------------------------------------------------#
# Prediction 1 point

dim_x1_data = len(x1_data)
k = np.zeros((dim_x1_data,dim_x1_data))
for i in range(dim_x1_data):
	for j in range(i+1):
		dx1 = x1_data[i] - x1_data[j]
		dx2 = x2_data[i] - x2_data[j]
		coef = - 0.5 * ( dx1**2 / gph_l1**2 + dx2**2 / gph_l2**2 )
		k[i,j] = gph_sigma_f * gph_sigma_f * math.exp( coef )
		k[j,i] = k[i,j]
		if i == j:
			k[i,j] = k[i,j] + sigma_n * sigma_n

x1_new = -7.0
x2_new = -5.0
k_new = np.zeros((dim_x1_data))
for i in range(dim_x1_data):
	dx1 = x1_new - x1_data[i]
	dx2 = x2_new - x2_data[i]
	coef = - 0.5 * ( dx1**2 / gph_l1**2 + dx2**2 / gph_l2**2 )
	k_new[i] = gph_sigma_f * gph_sigma_f * math.exp( coef )

k_inv = inv(k)
m1 = np.dot(k_new,k_inv)
y_new = np.dot(m1,y_data)

print("Prediction for 1 point: ")
print('x1_new, x2_new, y_new', x1_new, x2_new, y_new)

var_y = k[0,0] - k_new.dot(k_inv.dot(np.transpose(k_new)))
#print "var_y", var_y

x1 = np.append(x1_data,x1_new)
x2 = np.append(x2_data,x2_new)
y = np.append(y_data,y_new)

matplotlib_marker = []
for i in y:
	if i == -1.0: matplotlib_marker.append('o')
	if i == 1.0: matplotlib_marker.append('x')
	if i != -1.0 and i != 1.0: matplotlib_marker.append('d')

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.title('Prediction for a new point')

plt.xlabel('x1')
plt.ylabel('x2')

plt.grid(True)

print(len(x1), len(x2), len(matplotlib_marker))

for i in range(len(x1)):
	plt.scatter(x1[i], x2[i], marker=matplotlib_marker[i])

plt.savefig('gaussian_processes_2d_prediction_1_point.png', bbox_inches='tight')
#plt.show()
plt.close()

#----------------------------------------------------------------------------------------#
# Inference regular gridded data

x1 = np.arange(0,10.0,0.1)
x2 = np.arange(0,10.0,0.1)

dim_x1 = x1.shape[0]
dim_x2 = x2.shape[0]
dim_x = dim_x1 * dim_x2

Z = np.zeros((dim_x))
Z_var = np.zeros((dim_x))

i1_cp = 0
i2_cp = 0
for i in range(dim_x):
	x1_new = x1[i1_cp]
	x2_new = x2[i2_cp]
	k_new = np.zeros((dim_x1_data))
	for j in range(dim_x1_data):
		dx1 = x1_new - x1_data[j]
		dx2 = x2_new - x2_data[j]
		coef = - 0.5 * ( dx1**2 / gph_l1**2 + dx2**2 / gph_l2**2 )
		k_new[j] = gph_sigma_f * gph_sigma_f * math.exp( coef )
	k_inv = inv(k)
	m1 = np.dot(k_new,k_inv)
	y_new = np.dot(m1,y_data)
	Z[i] = y_new
	Z_var[i] = k[0,0] - k_new.dot(k_inv.dot(np.transpose(k_new)))
	i2_cp = i2_cp + 1
	if i2_cp == dim_x2:
		i1_cp = i1_cp + 1
		i2_cp = 0
	if i1_cp == dim_x1:
		i1_cp = 0

Z = np.reshape(Z, (dim_x1,dim_x2))
Z_var = np.reshape(Z_var, (dim_x1,dim_x2))

plt.imshow(Z.T, interpolation='bilinear', origin='lower')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian processes 2d regression')
plt.grid(True)
plt.savefig('gaussian_process_2d_gridded_data.png')
plt.close()

plt.imshow(Z_var.T, interpolation='bilinear', origin='lower')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian processes 2d regression (variance)')
plt.grid(True)
plt.savefig('gaussian_process_2d_gridded_data_var.png')
plt.close()

#----------------------------------------------------------------------------------------#
# Classification

for x in np.nditer(Z, op_flags=['readwrite']):
	x[...] = 1.0 / ( 1.0 + math.exp(-x))

plt.imshow(Z.T, interpolation='bilinear', origin='lower',vmin=0,vmax=1)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian processes 2d for classification')
plt.grid(True)
plt.savefig('gaussian_process_2d_gridded_data_normalized.png')
plt.close()

CS = plt.contour(x1, x2, Z.T, origin='lower')
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian processes 2d for classification (contour plot)')
plt.grid(True)
plt.savefig('gaussian_process_2d_gridded_data_normalized_contour_plot.png')
plt.close()
