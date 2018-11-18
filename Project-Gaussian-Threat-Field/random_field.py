import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def rf(shape, elements):
    # Creates a field of random gaussian elements
    # Inputs: shape = the size of the entire field in (x_length, y_length)
    #         elements = number of gaussian elements to be added to the field


    field = np.zeros(shape)
    width = shape[0]
    height = shape[1]
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y, indexing='ij', sparse='true')
    for i in range(elements): # for each element
        # select center of element
        x0 = np.random.randint(0, width)
        y0 = np.random.randint(0, height)
        print(x0, y0)
        # select spacing parameters, dynamic w/r to field size
        sigma_x = width/15 + np.random.rand()*width/20
        sigma_y = height/15 + np.random.rand()*height/20
        # select amplitude
        A = np.random.rand()*10
        # add to the field a gaussian surface with generated parameters
        field += A * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))

    return field

if __name__ == "__main__":
    field = rf((1080,1920), 1)
    #cmap = LinearSegmentedColormap.from_list('mycmap', ['#0D0887', '#C5407E', '#f0f724'])
    #plt.imshow(field, cmap=cmap, interpolation='nearest')
    plt.imshow(field, cmap=plt.get_cmap('plasma'), interpolation='nearest')
    plt.show()
