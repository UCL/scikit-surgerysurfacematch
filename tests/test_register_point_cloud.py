# -*- coding: utf-8 -*-

import cv2
import numpy as np

import sksurgerysurfacematch.utils.projection_utils as pu
import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sr
import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_reconstruction as reg


def test_point_cloud_registration():

    left_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.intrinsics.txt')
    right_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.distortion.txt')
    right_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.distortion.txt')
    l2r_matrix = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.l2r.txt')

    l2r_rmat = l2r_matrix[0:3, 0:3]
    l2r_tvec = l2r_matrix[0:3, 3:4]

    pointcloud_file = 'tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_CT_cut.xyz'
    point_cloud = np.loadtxt(pointcloud_file)

    model_to_camera = np.eye(4)

    left_image = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_left.bmp')
    left_undistorted = cv2.undistort(left_image, left_intrinsics, left_distortion)
    left_mask = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_MASK_eval_inverted.png')
    left_mask = cv2.cvtColor(left_mask, cv2.COLOR_BGR2GRAY)
    right_image = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_right.bmp')
    right_undistorted = cv2.undistort(right_image, right_intrinsics, right_distortion)

    # Produce picture of gold standard registration.
    pu.reproject_and_save(left_undistorted, model_to_camera, point_cloud, left_intrinsics,
                          output_file='tests/output/open_cas_tmi_gold.png')

    # Create registration pipeline.
    reg_points_to_vid = \
        reg.Register3DToStereoVideo(None,
                                    sr.SGBMReconstructor(),
                                    pir.RigidRegistration(),
                                    left_intrinsics,
                                    left_distortion,
                                    right_intrinsics,
                                    right_distortion,
                                    l2r_rmat,
                                    l2r_tvec,
                                    left_mask=left_mask,
                                    z_range=[45, 65],
                                    voxel_reduction=[5, 5, 5]
                                    )

    residual, registration = reg_points_to_vid.register(point_cloud,
                                                        left_undistorted,
                                                        right_undistorted,
                                                        model_to_camera
                                                        )

    print(f'Model: {pointcloud_file}')
    print(f'{len(point_cloud)} points in reference point cloud')

    print("Residual:" + str(residual))
    print("Registration, using single-view, with full point cloud:\n" + str(registration))

    pu.reproject_and_save(left_undistorted, registration, point_cloud, left_intrinsics,
                          output_file='tests/output/open_cas_tmi_registered.png')

    assert residual < 5.0

