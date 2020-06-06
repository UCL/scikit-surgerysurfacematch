# -*- coding: utf-8 -*-

""" Methods for saving .ply files etc. """

import os
import numpy as np


def write_ply(ply_data: list, ply_file: str):
    """
    Writes a .ply format file.

    :param ply_data: points and colours stored as list
    :param ply_file: file name
    """
    dir_name = os.path.dirname(ply_file)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    file = open(ply_file, "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar nx
property uchar ny
property uchar nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(ply_data), "".join(ply_data)))

    file.close()


def write_pointcloud(points: np.ndarray,
                     colours: np.ndarray,
                     file_name: str):
    """
    Write point cloud points and colours to .ply file.
    :param points: [Nx3] ndarray, of x, y, z coordinates
    :param colours: [Nx3] ndarray, of r, g, b colours
    :param file_name: filename including .ply extension
    """
    ply_data = []
    alpha = 0
    for j, _ in enumerate(points):
        ply_data.append("%f %f %f %d %d %d %d %d %d %d\n"%
                        (points[j][0], points[j][1], points[j][2],
                         1, 1, 1,
                         colours[j][0], colours[j][1], colours[j][2],
                         alpha))

    write_ply(ply_data, file_name)
