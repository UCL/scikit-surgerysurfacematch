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
                    left_mask: np.ndarray = None
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
        :return: [Nx6] point cloud where the 6 columns
        are x, y, z in left camera space, and r, g, b, colors.
        """
        # Has format X,Y,Z (3D triangulated point), x_left, y_left,
        # x_right, y_right (2D matches).
        points_stoyanov = \
            cvpy.reconstruct_points_using_stoyanov(left_image,
                                                   left_camera_matrix,
                                                   right_image,
                                                   right_camera_matrix,
                                                   left_to_right_rmat,
                                                   left_to_right_tvec,
                                                   self.use_hartley
                                                   )

        points_xyz = points_stoyanov[:, :3]
        left_matches_xy_points = points_stoyanov[:, 3:5]

        num_points = points_stoyanov.shape[0]

        rgb_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        if left_mask is not None:

            # Allocate the max required size, then we can trim it down at the
            # end.
            result = np.zeros((num_points, 6))

            i = 0
            for point_idx in range(0, num_points):
                x_l_c = int(left_matches_xy_points[point_idx][0])
                y_l_c = int(left_matches_xy_points[point_idx][1])

                if left_mask[y_l_c][x_l_c] > 0:

                    row = np.array([[points_xyz[point_idx][0],
                                     points_xyz[point_idx][1],
                                     points_xyz[point_idx][2],
                                     rgb_image[y_l_c][x_l_c][0],
                                     rgb_image[y_l_c][x_l_c][1],
                                     rgb_image[y_l_c][x_l_c][2]
                                     ]])
                    
                    result[i, :] = row
                    i += 1
            
            result = result[:i, :]

        else:

            result = np.zeros((num_points, 6))
            result[:, 0:3] = points_stoyanov[:, 0:3]

            for point_idx in range(0, num_points):
                x_c = int(points_stoyanov[point_idx][3])
                y_c = int(points_stoyanov[point_idx][4])
                result[point_idx][3] = rgb_image[y_c][x_c][0]
                result[point_idx][4] = rgb_image[y_c][x_c][1]
                result[point_idx][5] = rgb_image[y_c][x_c][2]

        return result
