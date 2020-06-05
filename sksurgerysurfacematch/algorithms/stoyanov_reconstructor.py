# -*- coding: utf-8 -*-

""" Surface reconstruction using Stoyanov MICCAI 2010 paper. """

import numpy as np
import cv2

import sksurgeryopencvpython as cvpy
import sksurgerypclpython as pclpy
import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr


class StoyanovReconstructor(sr.StereoReconstructor):
    """
    Constructor.
    """
    def __init__(self,
                 use_hartley=False,
                 use_voxel_grid_reduction=False,
                 voxel_grid_size=1,
                 use_statistical_outlier_removal=False,
                 outlier_mean_k=1,
                 outlier_std_dev=1.0
                 ):
        super(StoyanovReconstructor, self).__init__()
        self.use_hartley = use_hartley
        self.use_voxel_grid_reduction = use_voxel_grid_reduction
        self.voxel_grid_size = voxel_grid_size
        self.use_statistical_outlier_removal = use_statistical_outlier_removal
        self.outlier_mean_k = outlier_mean_k
        self.outlier_std_dev = outlier_std_dev

    def reconstruct(self,
                    left_image: np.ndarray,
                    right_image: np.ndarray,
                    left_camera_matrix: np.ndarray,
                    left_dist_coeffs: np.ndarray,
                    right_camera_matrix: np.ndarray,
                    right_dist_coeffs: np.ndarray,
                    left_to_right_rmat: np.ndarray,
                    left_to_right_tvec: np.ndarray
                    ):
        """
        Implementation of dense stereo surface reconstruction using
        Dan Stoyanov's MICCAI 2010 method.

        Camera parameters are those obtained from OpenCV.

        If distortion coefficients are not None, then image is undistorted
        first.

        :param left_image: left image, RGB
        :param right_image: right image, RGB
        :param left_camera_matrix: [3x3] camera matrix
        :param left_dist_coeffs: [1x5] distortion coefficients
        :param right_camera_matrix: [3x3] camera matrix
        :param right_dist_coeffs: [1x5] distortion coefficients
        :param left_to_right_rmat: [3x3] rotation matrix
        :param left_to_right_tvec: [3x1] translation vector
        :return: [Nx3] point cloud where the 3 columns
        are x, y, z in left camera space.
        """
        left_im = left_image
        if left_dist_coeffs is not None:
            left_im = cv2.undistort(left_image,
                                    left_camera_matrix,
                                    left_dist_coeffs)

        right_im = right_image
        if right_dist_coeffs is not None:
            right_im = cv2.undistort(right_image,
                                     right_camera_matrix,
                                     right_dist_coeffs)

        points_7 = cvpy.reconstruct_points_using_stoyanov(left_im,
                                                          left_camera_matrix,
                                                          right_im,
                                                          right_camera_matrix,
                                                          left_to_right_rmat,
                                                          left_to_right_tvec,
                                                          self.use_hartley
                                                          )

        points = points_7[:, 0:3]

        if self.use_voxel_grid_reduction:
            points = pclpy.down_sample_points(points,
                                              self.voxel_grid_size,
                                              self.voxel_grid_size,
                                              self.voxel_grid_size)

        if self.use_statistical_outlier_removal:
            points = pclpy.remove_outlier_points(points,
                                                 self.outlier_mean_k,
                                                 self.outlier_std_dev)

        return points
