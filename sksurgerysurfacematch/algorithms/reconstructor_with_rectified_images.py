# -*- coding: utf-8 -*-

""" Base class for surface reconstruction on already rectified images. """

import numpy as np
import cv2

import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr


class StereoReconstructorWithRectifiedImages(sr.StereoReconstructor):
    """
    Base class for those stereo reconstruction methods that work specifically
    from rectified images. This class handles rectification and
    the necessary coordinate transformations. Note: The client calls
    the reconstruct() method which requires undistorted images,
    which are NOT already rectified. It's THIS class that does the
    rectification for you, and calls through to the _compute_disparity()
    method that derived classes must implement.
    """
    def __init__(self,
                 lower_disparity_multiplier=0.75,
                 upper_disparity_multiplier=2.0):
        """
        Constructor creates some member variables, so this class
        becomes statefull. You call reconstruct() once, and then
        you can call extract multiple times with different masks
        to pull out different subsets of data.

        :param lower_disparity_multiplier: min=median - (this * std).
        :param upper_disparity_multiplier: max=median + (this * std).
        """
        super().__init__()
        self.disparity = None
        self.points = None
        self.rgb_image = None
        self.r_1 = None
        self.lower_disparity_multiplier = lower_disparity_multiplier
        self.upper_disparity_multiplier = upper_disparity_multiplier

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
        Implementation of stereo surface reconstruction that takes
        undistorted images, rectifies them, asks derived classes
        to compute a disparity map on the rectified images, and
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
        :param left_mask: mask image, single channel, same size as left_image
        :return: [Nx6] point cloud where the 6 columns
        are x, y, z in left camera space, followed by r, g, b colours.
        """
        # pylint:disable=too-many-locals
        (width, height) = (left_image.shape[1], left_image.shape[0])

        self.r_1, r_2, p_1, p_2, q_mat, _, _ = \
            cv2.stereoRectify(left_camera_matrix,
                              left_dist_coeffs,
                              right_camera_matrix,
                              right_dist_coeffs,
                              (width, height),
                              left_to_right_rmat,
                              left_to_right_tvec
                              )

        undistort_rectify_map_l_x, undistort_rectify_map_l_y = \
            cv2.initUndistortRectifyMap(left_camera_matrix,
                                        left_dist_coeffs,
                                        self.r_1, p_1,
                                        (width, height), cv2.CV_32FC1)

        undistort_rectify_map_r_x, undistort_rectify_map_r_y = \
            cv2.initUndistortRectifyMap(right_camera_matrix,
                                        right_dist_coeffs,
                                        r_2, p_2,
                                        (width, height), cv2.CV_32FC1)

        left_rectified = cv2.remap(left_image, undistort_rectify_map_l_x,
                                   undistort_rectify_map_l_y, cv2.INTER_LINEAR)

        right_rectified = cv2.remap(right_image, undistort_rectify_map_r_x,
                                    undistort_rectify_map_r_y, cv2.INTER_LINEAR)

        # Need to remap the mask if we have one
        if left_mask is not None:

            left_mask = cv2.remap(left_mask, undistort_rectify_map_l_x,
                                  undistort_rectify_map_l_y, cv2.INTER_NEAREST)

        self.disparity = self._compute_disparity(left_rectified,
                                                 right_rectified)

        print(f'Disparity: {self.disparity.shape}')

        self.points = cv2.reprojectImageTo3D(self.disparity, q_mat)
        self.rgb_image = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)

        # Calls method below to extract data.
        return self.extract(left_mask)

    def extract(self, left_mask: np.ndarray):
        """
        Extracts the actual point cloud. This is a separate method,
        so that you can reconstruct once using reconstruct(), and then
        call this extract method with multiple masks, without incurring
        the cost of multiple calls to the reconstruction algorithm, which
        may be expensive.
        :param left_mask: mask image, single channel, same size as left_image
        :return: [Nx6] point cloud where the 6 columns
        are x, y, z in left camera space, followed by r, g, b colours.
        """

        median_disp = np.median(self.disparity)
        std_dev_disp = np.std(self.disparity)
        lower_bound = (median_disp - (self.lower_disparity_multiplier
                                              * std_dev_disp))
        upper_bound = (median_disp + (self.upper_disparity_multiplier
                                              * std_dev_disp))
        mask = np.logical_and(self.disparity > lower_bound,
                              self.disparity < upper_bound)

        print("Disparity, min=" + str(self.disparity.min())
              + ", max=" + str(self.disparity.max())
              + ", med=" + str(np.median(self.disparity))
              + ", std=" + str(np.std(self.disparity))
              + ", lower=" + str(lower_bound)
              + ", upper=" + str(upper_bound)
              )

        if left_mask is not None:

            if left_mask.dtype != bool:
                left_mask = left_mask > 0

            mask = np.logical_and(mask, left_mask)

        out_points = self.points[mask]
        out_colors = self.rgb_image[mask]

        non_zero = np.count_nonzero(mask)
        result = np.zeros((non_zero, 6))

        result[:, 0:3] = out_points
        result[:, 3:6] = out_colors

        # Convert from first (left) camera rectified to left camera unrectified
        result[:, 0:3] = np.transpose(
            np.matmul(np.linalg.inv(self.r_1), np.transpose(result[:, 0:3])))

        return result

    def _compute_disparity(self, left_rectified_image, right_rectified_image):
        """
        Derived classes implement this to compute a disparity map from
        pre-rectified images. But clients still call the reconstruct() method.

        The returned disparity map, must be equivalent to what OpenCV
        returns from other stereo reconstructors like the SGBM reconstructor.
        That is an image, same size as left and right rectified images,
        of type float32, where each pixel value represents left-to-right
        disparity.

        :param left_rectified_image: undistorted, rectified image, BGR
        :param right_rectified_image: undistorted, rectified image, BGR
        :return: disparity map
        """
        raise NotImplementedError("Derived classes should implement this.")
