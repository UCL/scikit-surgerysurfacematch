# -*- coding: utf-8 -*-

""" Surface reconstruction using Stoyanov MICCAI 2010 paper. """

import numpy as np
import cv2

import sksurgeryopencvpython as cvpy
import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr


class StoyanovReconstructor(sr.StereoReconstructor):
    """
    Constructor.
    """
    def __init__(self, use_hartley=False):
        super(StoyanovReconstructor, self).__init__()
        self.use_hartley = use_hartley

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
        Implementation of dense stereo surface reconstruction using
        Dan Stoyanov's MICCAI 2010 method.

        Camera parameters are those obtained from OpenCV.

        :param left_image: undistorted left image, BGR
        :param left_camera_matrix: [3x3] camera matrix
        :param left_dist_coeffs: [1xN] distortion coefficients
        :param right_image: undistorted right image, BGR
        :param right_camera_matrix: [3x3] camera matrix
        :param right_dist_coeffs: [1xN] distortion coefficients
        :param left_to_right_rmat: [3x3] rotation matrix
        :param left_to_right_tvec: [3x1] translation vector
        :param left_mask: mask image, single channel, same size as left_image
        :param right_mask: mask image, single channel, same size as right_image
        :return: [Nx6] point cloud where the 6 columns
        are x, y, z in left camera space, and r, g, b, colors.
        """
        points_7 = cvpy.reconstruct_points_using_stoyanov(left_image,
                                                          left_camera_matrix,
                                                          right_image,
                                                          right_camera_matrix,
                                                          left_to_right_rmat,
                                                          left_to_right_tvec,
                                                          self.use_hartley
                                                          )

        rgb_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        if left_mask is not None and right_mask is not None:

            result = np.zeros((0, 6))

            for point_counter in range(0, points_7.shape[0]):
                x_l_c = int(points_7[point_counter][3])
                y_l_c = int(points_7[point_counter][4])
                x_r_c = int(points_7[point_counter][5])
                y_r_c = int(points_7[point_counter][6])
                if left_mask[y_l_c][x_l_c] > 0 \
                        and right_mask[y_r_c][x_r_c] > 0:
                    row = np.array([[points_7[point_counter][0],
                                     points_7[point_counter][1],
                                     points_7[point_counter][2],
                                     rgb_image[y_l_c][x_l_c][0],
                                     rgb_image[y_l_c][x_l_c][1],
                                     rgb_image[y_l_c][x_l_c][2]
                                     ]])
                    result = np.append(result, row, axis=0)

        else:

            result = np.zeros((points_7.shape[0], 6))
            result[:, 0:3] = points_7[:, 0:3]

            for point_counter in range(0, points_7.shape[0]):
                x_c = int(points_7[point_counter][3])
                y_c = int(points_7[point_counter][4])
                result[point_counter][3] = rgb_image[y_c][x_c][0]
                result[point_counter][4] = rgb_image[y_c][x_c][1]
                result[point_counter][5] = rgb_image[y_c][x_c][2]

        return result
