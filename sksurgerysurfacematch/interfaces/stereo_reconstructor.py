# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for classes that do stereo recon. """

import numpy as np


class StereoReconstructor:
    """
    Base class for stereo reconstruction algorithms. Clients call
    the reconstruct() method, passing in undistorted images.
    The output is an [Nx6] array where the N rows are each point,
    and the 6 columns are x, y, z, r, g, b.
    """
    # pylint:disable=too-many-arguments
    def reconstruct(self,
                    left_image: np.ndarray,
                    left_camera_matrix: np.ndarray,
                    left_dist_coeffs: np.ndarray,
                    right_image: np.ndarray,
                    right_camera_matrix: np.ndarray,
                    right_dist_coeffs: np.ndarray,
                    left_to_right_rmat: np.ndarray,
                    left_to_right_tvec: np.ndarray,
                    left_mask: np.ndarray = None,
                    right_mask: np.ndarray = None
                    ):
        """
        A derived class must implement this.
        Camera parameters are those obtained from OpenCV.

        :param left_image: left image, BGR
        :param left_camera_matrix: [3x3] camera matrix
        :param left_dist_coeffs: [1xN] distortion coefficients
        :param right_image: right image, BGR
        :param right_camera_matrix: [3x3] camera matrix
        :param right_dist_coeffs: [1xN] distortion coefficients
        :param left_to_right_rmat: [3x3] rotation matrix
        :param left_to_right_tvec: [3x1] translation vector
        :param left_mask: mask image, single channel, same size as left_image
        :param right_mask: mask image, single channel, same size as right_image
        :return: [Nx6] point cloud in left camera space, where N is the number
        of points, and 6 columns are x,y,z,r,g,b.
        """
        raise NotImplementedError("Derived classes should implement this.")
