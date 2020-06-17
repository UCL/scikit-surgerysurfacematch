# -*- coding: utf-8 -*-

import cv2
import numpy as np

import sksurgerysurfacematch.utils.projection_utils as pu
import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sr
import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_mosaic as sm


def test_point_cloud_mosaiicing():

    left_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.intrinsics.txt')
    right_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.distortion.txt')
    right_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.distortion.txt')
    l2r_matrix = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.l2r.txt')

    l2r_rmat = l2r_matrix[0:3, 0:3]
    l2r_tvec = l2r_matrix[0:3, 3:4]

    left_image_t0 = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_left.bmp')
    left_undistorted_t0 = cv2.undistort(left_image_t0, left_intrinsics, left_distortion)
    left_mask_t0 = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_MASK_eval_inverted.png')
    left_mask_t0 = cv2.cvtColor(left_mask_t0, cv2.COLOR_BGR2GRAY)
    right_image_t0 = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_right.bmp')
    right_undistorted_t0 = cv2.undistort(right_image_t0, right_intrinsics, right_distortion)

    video_mosaiicer = sm.Register3DToMosaicedStereoVideo(None,
                                                         sr.SGBMReconstructor(),
                                                         None,
                                                         left_intrinsics,
                                                         left_distortion,
                                                         right_intrinsics,
                                                         right_distortion,
                                                         l2r_rmat,
                                                         l2r_tvec,
                                                         left_mask=left_mask_t0)

    video_mosaiicer.grab(left_undistorted_t0,
                         right_undistorted_t0)

    video_mosaiicer.grab(left_undistorted_t0,
                         right_undistorted_t0)

    assert True
