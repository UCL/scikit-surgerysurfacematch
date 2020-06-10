
import cv2
import numpy as np

import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sr
import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.algorithms.goicp_registration as gir

import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_reconstruction \
    as reg

import sksurgeryvtk.widgets.vtk_overlay_window as ow

def test_point_cloud_registration():

    pointcloud = np.loadtxt('tests/data/synthetic_liver/liver-H07.xyz')

    left_image = cv2.imread('tests/data/synthetic_liver/synthetic-left.png')
    left_mask = cv2.imread('tests/data/synthetic_liver/synthetic-left-mask.png')
    left_mask = cv2.cvtColor(left_mask, cv2.COLOR_BGR2GRAY)
    right_image = cv2.imread('tests/data/synthetic_liver/synthetic-right.png')
    left_intrinsics = np.loadtxt('tests/data/synthetic_liver/calib/calib.left.intrinsics.txt')
    right_intrinsics = np.loadtxt('tests/data/synthetic_liver/calib/calib.right.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/synthetic_liver/calib/calib.left.distortion.txt')
    right_distortion = np.loadtxt('tests/data/synthetic_liver/calib/calib.right.distortion.txt')
    l2r_matrix = np.loadtxt('tests/data/synthetic_liver/calib/calib.l2r.txt')

    left_mask = cv2.imread('tests/data/synthetic_liver/mask-left.png', 0)

    left_mask = cv2.threshold(left_mask, 127, 255, cv2.THRESH_BINARY)[1]

    model_to_world = np.loadtxt('tests/data/synthetic_liver/model_to_world.txt')
    camera_to_world = np.loadtxt('tests/data/synthetic_liver/camera_to_world.txt')

    camera_to_model = np.linalg.inv(model_to_world) @ camera_to_world

    l2r_rmat = l2r_matrix[:3, :3]
    l2r_tvec = l2r_matrix[3, :]
    
    print(f'{len(pointcloud)} points in point cloud')

    reg_points_to_vid = \
        reg.Register3DToStereoVideo(None,
                                    sr.SGBMReconstructor(),
                                    pir.RigidRegistration(),
                                    left_mask=left_mask,
                                    voxel_reduction=[5, 5, 5],
                                    statistical_outlier_reduction=[500, 5]
                                    )

    residual, registration = reg_points_to_vid.register(pointcloud,
                                                        left_image,
                                                        left_intrinsics,
                                                        left_distortion,
                                                        right_image,
                                                        right_intrinsics,
                                                        right_distortion,
                                                        l2r_rmat,
                                                        l2r_tvec)
    print(residual)
    print(registration)
