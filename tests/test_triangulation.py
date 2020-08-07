#  -*- coding: utf-8 -*-

""" Tests for triangulation. """

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc
import sksurgeryopencvpython as cvcpp


def test_triangulation():
    """
    Testing because I need to understand the difference between triangulating
    unrectified images, and rectified images.
    """

    left_images = []
    files = glob.glob('tests/data/chessboard/left/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        left_images.append(image)
    assert(len(left_images) == 9)

    right_images = []
    files = glob.glob('tests/data/chessboard/right/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        right_images.append(image)
    assert (len(right_images) == 9)

    chessboard_detector = cpd.ChessboardPointDetector((14, 10),
                                                      3,
                                                      (1, 1)
                                                      )

    calibrator = \
        sc.StereoVideoCalibrationDriver(chessboard_detector,
                                        chessboard_detector,
                                        140)

    # Repeatedly grab data, until you have enough.
    for i, _ in enumerate(left_images):
        success_l, success_r =  \
            calibrator.grab_data(left_images[i], right_images[i])
        assert success_l > 0

    # Then do calibration
    reproj_err, recon_err, params = calibrator.calibrate()

    left_image = cv2.imread('tests/data/chessboard/left-2520.png')
    right_image = cv2.imread('tests/data/chessboard/right-2520.png')
    left_intrinsics = params.left_params.camera_matrix
    left_distortion = params.left_params.dist_coeffs
    right_intrinsics = params.right_params.camera_matrix
    right_distortion = params.right_params.dist_coeffs
    l2r_rmat = params.l2r_rmat
    l2r_tvec = params.l2r_tvec

    left_undistorted = cv2.undistort(left_image, left_intrinsics, left_distortion)
    right_undistorted = cv2.undistort(right_image, right_intrinsics, right_distortion)

    pd = cpd.ChessboardPointDetector((14, 10), 3)
    l_ud_ids, l_ud_obj, l_ud_im = pd.get_points(left_undistorted)
    r_ud_ids, r_ud_obj, r_ud_im = pd.get_points(right_undistorted)

    w = left_image.shape[1]
    h = left_image.shape[0]
    R1, R2, P1, P2, Q, vp1, vp2 = cv2.stereoRectify(left_intrinsics,
                                                    left_distortion,
                                                    right_intrinsics,
                                                    right_distortion,
                                                    (w, h),
                                                    l2r_rmat,
                                                    l2r_tvec
                                                    )

    undistort_rectify_map_l_x, undistort_rectify_map_l_y = \
        cv2.initUndistortRectifyMap(left_intrinsics, left_distortion, R1, P1, (w, h), cv2.CV_32FC1)

    undistort_rectify_map_r_x, undistort_rectify_map_r_y = \
        cv2.initUndistortRectifyMap(right_intrinsics, right_distortion, R2, P2, (w, h), cv2.CV_32FC1)

    left_rectified = cv2.remap(left_image, undistort_rectify_map_l_x,
                               undistort_rectify_map_l_y, cv2.INTER_LINEAR)

    right_rectified = cv2.remap(right_image, undistort_rectify_map_r_x,
                                undistort_rectify_map_r_y, cv2.INTER_LINEAR)

    l_rf_ids, l_rf_obj, l_rf_im = pd.get_points(left_rectified)
    r_rf_ids, r_rf_obj, r_rf_im = pd.get_points(right_rectified)

    points_4D = cv2.triangulatePoints(P1, P2, np.transpose(l_rf_im), np.transpose(r_rf_im))
    points_4D = np.transpose(points_4D)
    points_4D = points_4D[:, 0:-1] / points_4D[:, -1].reshape((140, 1))

    # Convert from first (left) camera rectified to left camera unrectified
    points_3D = np.transpose(np.matmul(np.linalg.inv(R1), np.transpose(points_4D)))

    # Triangulate points in undistorted, but not rectified space.
    image_points = np.zeros((l_ud_im.shape[0], 4))
    image_points[:, 0:2] = l_ud_im
    image_points[:, 2:4] = r_ud_im

    triangulated = cvcpp.triangulate_points_using_hartley(image_points,
                                                          left_intrinsics,
                                                          right_intrinsics,
                                                          l2r_rmat,
                                                          l2r_tvec)
    # All being well, points_3D == triangulated
    assert np.allclose(points_3D, triangulated, rtol=0.1, atol=0.1)
