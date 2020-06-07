# -*- coding: utf-8 -*-

""" Base class for surface reconstruction on already rectified images. """

import numpy as np
import cv2

import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr


class StereoReconstructorWithRectifiedImages(sr.StereoReconstructor):
    """
    Constructor.
    """
    def __init__(self):
        super(StereoReconstructorWithRectifiedImages, self).__init__()

    def reconstruct(self,
                    left_image: np.ndarray,
                    left_camera_matrix: np.ndarray,
                    left_dist_coeffs: np.ndarray,
                    right_image: np.ndarray,
                    right_camera_matrix: np.ndarray,
                    right_dist_coeffs: np.ndarray,
                    left_to_right_rmat: np.ndarray,
                    left_to_right_tvec: np.ndarray
                    ):
        """
        Implementation of stereo surface reconstruction that takes
        undistorted images, rectifies them, asks derived classes
        to compute a disparity map on rectified images, and
        then sorts out extracting points and their colours.

        Camera parameters are those obtained from OpenCV.

        :param left_image: undistorted left image, BGR
        :param left_camera_matrix: [3x3] camera matrix
        :param left_dist_coeffs: [1xN] distortion coefficients
        :param right_image: undistorted right image, BGR
        :param right_camera_matrix: [3x3] camera matrix
        :param right_dist_coeffs: [1xN] distortion coefficients
        :param left_to_right_rmat: [3x3] rotation matrix
        :param left_to_right_tvec: [3x1] translation vector
        :return: [Nx3] point cloud where the 3 columns
        are x, y, z in left camera space.
        """
        w = left_image.shape[1]
        h = left_image.shape[0]
        R1, R2, P1, P2, Q, vp1, vp2 = cv2.stereoRectify(left_camera_matrix,
                                                        left_dist_coeffs,
                                                        right_camera_matrix,
                                                        right_dist_coeffs,
                                                        (w, h),
                                                        left_to_right_rmat,
                                                        left_to_right_tvec
                                                        )

        undistort_rectify_map_l_x, undistort_rectify_map_l_y = \
            cv2.initUndistortRectifyMap(left_camera_matrix, left_dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)

        undistort_rectify_map_r_x, undistort_rectify_map_r_y = \
            cv2.initUndistortRectifyMap(right_camera_matrix, right_dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

        left_rectified = cv2.remap(left_image, undistort_rectify_map_l_x,
                                   undistort_rectify_map_l_y, cv2.INTER_LINEAR)

        right_rectified = cv2.remap(right_image, undistort_rectify_map_r_x,
                                    undistort_rectify_map_r_y, cv2.INTER_LINEAR)

        disparity = self._compute_disparity(left_rectified, right_rectified)

        points = cv2.reprojectImageTo3D(disparity, Q)
        rgb_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        mask = disparity > disparity.min()
        out_points = points[mask]
        out_colors = rgb_image[mask]

        non_zero = np.count_nonzero(mask)
        result = np.zeros((non_zero, 6))

        result[:, 0:3] = out_points
        result[:, 3:6] = out_colors

        # Convert from first (left) camera rectified to left camera unrectified
        result[:, 0:3] = np.transpose(np.matmul(np.linalg.inv(R1), np.transpose(result[:, 0:3])))

        return result

    def _compute_disparity(self, left_rectified_image, right_rectified_image):
        """
        Derived classes implement this to compute a disparity map from
        pre-rectified images. But clients still call the reconstruct method.

        (i.e. think of this method as 'protected'.)

        :param left_rectified_image: undistorted, rectified image, BGR
        :param right_rectified_image: undistorted, rectified image, BGR
        :return: disparity map
        """
        raise NotImplementedError("Derived classes should implement this.")


