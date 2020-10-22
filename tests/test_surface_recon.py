#  -*- coding: utf-8 -*-

""" Tests for default surface recon. """
import pytest
import numpy as np
import cv2
import sksurgerypclpython as pclp
import sksurgerysurfacematch.algorithms.stoyanov_reconstructor as sr
import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sgbm
import sksurgerysurfacematch.utils.ply_utils as pl


def test_stoyanov_and_sgbm():

    # Example from 2nd silicon heart phantom dataset from Hamlyn. http://hamlyn.doc.ic.ac.uk/vision/.
    # Technically, we should undistort the image first before reconstructing.
    left_intrinsics_file = 'tests/data/stoyanov/calib.left.intrinsic.txt'
    left_intrinsics = np.loadtxt(left_intrinsics_file)

    left_distortion_file = 'tests/data/stoyanov/calib.left.distortion.txt'
    left_dist_coeffs = np.loadtxt(left_distortion_file)

    right_intrinsics_file = 'tests/data/stoyanov/calib.right.intrinsic.txt'
    right_intrinsics = np.loadtxt(right_intrinsics_file)

    right_distortion_file = 'tests/data/stoyanov/calib.right.distortion.txt'
    right_dist_coeffs = np.loadtxt(right_distortion_file)

    l2r_file = 'tests/data/stoyanov/calib.l2r.4x4'
    l2r = np.loadtxt(l2r_file)

    rotation_matrix = l2r[0:3, 0:3]
    translation_vector = l2r[0:3, 3:4]

    left_image = cv2.imread('tests/data/stoyanov/f7_dynamic_deint_L_0100.png')
    right_image = cv2.imread('tests/data/stoyanov/f7_dynamic_deint_R_0100.png')

    # I don't yet know why OpenCV tends to read a grey scale image as BGR.
    mask_image = cv2.imread('tests/data/stoyanov/f7_dynamic_deint_mask_0100.png')
    if len(mask_image.shape) > 2:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    left_undistorted = cv2.undistort(left_image, left_intrinsics, left_dist_coeffs)
    right_undistorted = cv2.undistort(right_image, right_intrinsics, right_dist_coeffs)

    reconstructor = sr.StoyanovReconstructor()

    points = reconstructor.reconstruct(left_undistorted,   # should be undistorted, but doesn't need to be rectified
                                       left_intrinsics,
                                       right_undistorted,  # should be undistorted, but doesn't need to be rectified
                                       right_intrinsics,
                                       rotation_matrix,
                                       translation_vector,
                                       None)

    pl.write_pointcloud(points[:, 0:3], points[:, 3:6], 'tests/output/stoyanov.ply')

    assert points.shape[1] == 6
    assert points.shape[0] == 58244

    voxel_reduced_surface = pclp.down_sample_points(points[:, 0:3], 2, 2, 2)
    pl.write_pointcloud(voxel_reduced_surface[:, 0:3], np.ones((voxel_reduced_surface.shape[0], 3)) * 255,
                        'tests/output/stoyanov_grid_reduced.ply')
    print("Stoyanov, grid reduced cloud=" + str(voxel_reduced_surface.shape))

    outlier_reduced_surface = pclp.remove_outlier_points(voxel_reduced_surface, 10, 1)
    pl.write_pointcloud(outlier_reduced_surface[:, 0:3], np.ones((outlier_reduced_surface.shape[0], 3)) * 255,
                        'tests/output/stoyanov_outlier_reduced.ply')
    print("Stoyanov, outlier reduced cloud=" + str(outlier_reduced_surface.shape))

    # Now try Stoyanov masked version.
    points = reconstructor.reconstruct(left_undistorted,   # should be undistorted, but doesn't need to be rectified
                                       left_intrinsics,
                                       right_undistorted,  # should be undistorted, but doesn't need to be rectified
                                       right_intrinsics,
                                       rotation_matrix,
                                       translation_vector,
                                       mask_image)         # MUST be single channel

    pl.write_pointcloud(points[:, 0:3], points[:, 3:6], 'tests/output/stoyanov_masked.ply')

    assert points.shape[1] == 6
    assert points.shape[0] == 5253

    reconstructor = sgbm.SGBMReconstructor()
    points = reconstructor.reconstruct(left_undistorted,
                                       left_intrinsics,
                                       right_undistorted,
                                       right_intrinsics,
                                       rotation_matrix,
                                       translation_vector,
                                       left_mask=None)

    pl.write_pointcloud(points[:, 0:3], points[:, 3:6], 'tests/output/sgbm.ply')
    print("SGBM, cloud=" + str(points.shape))

    assert points.shape[1] == 6
    assert 104000 >= points.shape[0] >= 102000 # Slightly different answers between windows and mac/linux

    points = reconstructor.reconstruct(left_undistorted,
                                       left_intrinsics,
                                       right_undistorted,
                                       right_intrinsics,
                                       rotation_matrix,
                                       translation_vector,
                                       left_mask=mask_image)

    pl.write_pointcloud(points[:, 0:3], points[:, 3:6], 'tests/output/sgbm_masked.ply')

    assert points.shape[1] == 6
    assert 8000 >= points.shape[0] >= 6000

    print("SGBM masked, cloud=" + str(points.shape))
