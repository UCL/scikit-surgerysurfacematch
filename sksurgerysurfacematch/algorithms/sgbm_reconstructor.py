# -*- coding: utf-8 -*-

""" Surface reconstruction using OpenCV's SGBM reconstruction """

import numpy as np
import cv2

import sksurgerysurfacematch.algorithms.\
    reconstructor_with_rectified_images as sr


class SGBMReconstructor(sr.StereoReconstructorWithRectifiedImages):
    """
    Constructor. See OpenCV StereoSGBM for parameter comments.
    """
    def __init__(self,
                 min_disparity=16,
                 num_disparities=112,
                 block_size=3,
                 p_1=360,  # See Zhang 2019, DOI:10.1007/s11548-019-01974-6
                 p_2=1440, # See Zhang 2019, DOI:10.1007/s11548-019-01974-6
                 disp_12_max_diff=0,
                 uniqueness_ratio=0,
                 speckle_window_size=0,
                 speckle_range=0):
        super(SGBMReconstructor, self).__init__()
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p_1,
            P2=p_2,
            disp12MaxDiff=disp_12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range
        )

    def _compute_disparity(self, left_rectified_image, right_rectified_image):
        """
        Uses OpenCV's StereoSGBM to compute a disparity map from
        two, already rectified (done in base class) images.

        :param left_rectified_image: undistorted, rectified image, BGR
        :param right_rectified_image: undistorted, rectified image, BGR
        :return: disparity map
        """
        disparity = self.stereo.compute(
            left_rectified_image,
            right_rectified_image).astype(np.float32) / 16.0
        return disparity
