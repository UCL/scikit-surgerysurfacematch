# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for classes to do stereo recon. """

import numpy as np


class StereoReconstructor:
    """
    Constructor.
    """
    def __init__(self):
        return

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
        A derived class must implement this.
        Camera parameters are those obtained from OpenCV.

        :param left_image: left image, RGB
        :param right_image: right image, RGB
        :param left_camera_matrix: [3x3] camera matrix
        :param left_dist_coeffs: [1x5] distortion coefficients
        :param right_camera_matrix: [3x3] camera matrix
        :param right_dist_coeffs: [1x5] distortion coefficients
        :param left_to_right_rmat: [3x3] rotation matrix
        :param left_to_right_tvec: [3x1] translation vector
        :return: [Nx3] point cloud in left camera space
        """
        raise NotImplementedError("Derived classes should implement this.")
