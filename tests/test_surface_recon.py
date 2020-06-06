#  -*- coding: utf-8 -*-

""" Tests for default surface recon. """
import pytest
import numpy as np
import cv2
import sksurgerypclpython as pclp
import sksurgerysurfacematch.algorithms.stoyanov_reconstructor as sr
import sksurgerysurfacematch.utils.ply_utils as pl


def test_stoyanov_2010():
    """ Example test, as the code is unit tested in original project. """

    # Example from 2nd silicon heart phantom dataset from Hamlyn. http://hamlyn.doc.ic.ac.uk/vision/.
    # Technically, we should undistort the image first before reconstructing.
    left_intrinsics_file = 'tests/data/stoyanov/calib.left.intrinsic.txt'
    left_intrinsics = np.loadtxt(left_intrinsics_file)

    right_intrinsics_file = 'tests/data/stoyanov/calib.right.intrinsic.txt'
    right_intrinsics = np.loadtxt(right_intrinsics_file)

    l2r_file = 'tests/data/stoyanov/calib.l2r.4x4'
    l2r = np.loadtxt(l2r_file)

    rotation_matrix = l2r[0:3, 0:3]
    translation_vector = l2r[0:3, 3:4]

    left_image = cv2.imread('tests/data/stoyanov/f7_dynamic_deint_L_0100.png')
    right_image = cv2.imread('tests/data/stoyanov/f7_dynamic_deint_R_0100.png')

    reconstructor = sr.StoyanovReconstructor()

    points = reconstructor.reconstruct(left_image,
                                       right_image,
                                       left_intrinsics,
                                       right_intrinsics,
                                       rotation_matrix,
                                       translation_vector)
    assert points.shape[1] == 6
    assert points.shape[0] == 58244

    pl.write_pointcloud(points[:, 0:3], points[:, 3:6], 'tests/output/stoyanov.ply')

    voxel_reduced_surface = pclp.down_sample_points(points[:, 0:3], 2, 2, 2)
    pl.write_pointcloud(voxel_reduced_surface[:, 0:3], np.ones((voxel_reduced_surface.shape[0], 3)) * 255, 'tests/output/stoyanov_grid_reduced.ply')
    print("Grid reduced cloud=" + str(voxel_reduced_surface.shape))

    outlier_reduced_surface = pclp.remove_outlier_points(voxel_reduced_surface, 10, 1)
    pl.write_pointcloud(outlier_reduced_surface[:, 0:3], np.ones((outlier_reduced_surface.shape[0], 3)) * 255, 'tests/output/stoyanov_outlier_reduced.ply')
    print("Outlier reduced cloud=" + str(outlier_reduced_surface.shape))
