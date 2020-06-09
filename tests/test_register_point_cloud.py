
import cv2
import numpy as np

import sksurgerysurfacematch.algorithms.stoyanov_reconstructor as sr
import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_reconstruction \
    as reg


def test_point_cloud_registration():

    reg_points_to_vid = \
        reg.Register3DToStereoVideo(None,
                                    sr.StoyanovReconstructor(),
                                    pir.RigidRegistration(),
                                    None, None, None
                                    )

    pointcloud = np.loadtxt('tests/data/synthetic_liver/liver-H07.xyz')
    left_image = cv2.imread('tests/data/synthetic_liver/synthetic-left.png')
    right_image = cv2.imread('tests/data/synthetic_liver/synthetic-right.png')
    left_intrinsics = np.loadtxt('tests/data/synthetic_liver/calib/calib.left.intrinsics.txt')
    right_intrinsics = np.loadtxt('tests/data/synthetic_liver/calib/calib.right.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/synthetic_liver/calib/calib.left.distortion.txt')
    right_distortion = np.loadtxt('tests/data/synthetic_liver/calib/calib.right.distortion.txt')
    l2r_matrix = np.loadtxt('tests/data/synthetic_liver/calib/calib.l2r.txt')

    l2r_rmat = l2r_matrix[:3, :3]
    l2r_rvec = (cv2.Rodrigues(l2r_rmat))[0]
    l2r_tvec = l2r_matrix[3, :]

    pointcloud = pointcloud[::5, :]
    
    print(f'{len(pointcloud)} points in point cloud')

    residual, registration = reg_points_to_vid.register(pointcloud,
                                                        left_image,
                                                        left_intrinsics,
                                                        left_distortion,
                                                        right_image,
                                                        right_intrinsics,
                                                        right_distortion,
                                                        l2r_rmat,
                                                        l2r_tvec,
                                                        None)
    print(residual)
    print(registration)
