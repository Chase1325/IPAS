from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


def get_average_color_of_frame(file, offset=(), size=(40, 40)):
    """
    Given an open file of an image, crop to offset and size 
    and then find the average color of cropped image.
    """
    image = Image.open(file)  # open image

    width = image.size[0]
    height = image.size[1]

    if not offset: 
        offset = (
            width // 2 - size[0] // 2,
            height // 2 - size[1] // 2
        )

    image = image.crop((*offset, offset[0]+size[0],offset[1]+size[1]))

    im2arr = np.array(image)
    mean = [np.mean(im2arr[:,:,0]), np.mean(im2arr[:,:,1]), np.mean(im2arr[:,:,2])]
    return mean


def rgb_to_threat(im_rgb, colormap='gray'):
    if colormap == 'gray':
        colmap = []
        for i in range(256):
            colmap.append([i]*3)
    else:
        colmap = cm.get_cmap(plt.get_cmap(colormap))
        colmap = copy.deepcopy(colmap.colors)
        for i in range(len(colmap)):  # convert RGB percentages to values
            for j in range(len(colmap[i])):
                colmap[i][j] = round(colmap[i][j]*255)
    match = best_match(im_rgb, colmap)
    threat_value = colmap.index(match)
    return threat_value  # threat value is equal to the index of the closest match of im_rgb in the colmap array (0-255)


def distance(color1, color2):
    return math.sqrt(sum([(e1 - e2) ** 2 for e1, e2 in zip(color1, color2)]))


def best_match(sample, colors):
    by_distance = sorted(colors, key=lambda c: distance(c, sample))
    return by_distance[0]


if __name__ == "__main__":
    color_val = get_average_color_of_frame('test.png')
    print(color_val)
    threat = rgb_to_threat(color_val, 'gray')
    print(threat)
