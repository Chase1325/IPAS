import image_slicer
import numpy
import sys
from PIL import Image
import os

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def split_and_skew(file, projectors, width1, width2, d):
    tiles = image_slicer.slice(file, projectors, save=False)
    c = 0
    reduction = ((width2/width1)-1)/2

    for tile in tiles:
        img = tile.image
        img = img.rotate(90, expand=True)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if c % 2 == 0:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size
        new_width = round(reduction*width)
        m = -0.5
        #xshift = abs(m) * width
        coeffs = find_coeffs(
            [(new_width, 0), (width-new_width, 0), (width, height), (0, height)],
            [(0, 0), (width, 0), (width, height), (0, height)],
        )

        img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        img.save(f"{d}/{c}.png")
        c += 1


if __name__ == '__main__':
    d = '/home/roger/share'
    f = d + '/my_fig.png'
    f_new = d + '/my_fig_flip.png'
    img = Image.open(f)
    img = img.rotate(90, expand=True)
    img.save(f_new)
    img.close()
    split_and_skew(f_new, 2, 9.5, 10, d)

