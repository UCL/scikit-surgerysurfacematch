
import cv2
import numpy as np

import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sr
import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir

import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_reconstruction \
    as reg

def reproject_and_save(image,
                       camera_to_model, 
                       pointcloud, 
                       intrinsics, 
                       outfile):

    """Project model points back to 2D and save image """
    rmat = camera_to_model[:3, :3]
    rvec = cv2.Rodrigues(rmat)[0]
    tvec = camera_to_model  [:3, 3]

    projected, _ = cv2.projectPoints(pointcloud,
                    rvec,
                    tvec,
                    intrinsics,
                    None)

    for i in range(projected.shape[0]):

        x, y = projected[i][0]
        x = int(x)
        y = int(y)

        # Skip points that aren't in the bounds of image
        if 0 > x or x >= image.shape[1]:
            continue
        if 0 > y or y >= image.shape[0]:
            continue

        image[y, x, :] = [255, 0, 0]

    cv2.imwrite(outfile, image)

def test_point_cloud_registration():

    full_pointcloud = 'tests/data/synthetic_liver/liver-H07.xyz'
    front_surface_only_pointcloud = \
        'tests/data/synthetic_liver/liver-H07-reduced.xyz'

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

    l2r_rmat = l2r_matrix[:3, :3]
    l2r_tvec = l2r_matrix[3, :]
    
    model_to_world = np.loadtxt('tests/data/synthetic_liver/model_to_world.txt')
    camera_to_world = np.loadtxt('tests/data/synthetic_liver/camera_to_world.txt')


    model_to_camera = camera_to_world @ np.linalg.inv(model_to_world)
    camera_to_mod = np.linalg.inv(model_to_world) @ camera_to_world

    reg_points_to_vid = \
        reg.Register3DToStereoVideo(None,
                                    sr.SGBMReconstructor(),
                                    pir.RigidRegistration(),
                                    left_mask=left_mask,
                                    voxel_reduction=[5, 5, 5],
                                    statistical_outlier_reduction=[500, 5]
                                    )

    # Run pipeline using entire liver pointcloud (lots of points are not visible
    # in stere image)
    pointcloud = np.loadtxt(full_pointcloud)
    
    residual, registration = reg_points_to_vid.register(pointcloud,
                                                        left_image,
                                                        left_intrinsics,
                                                        left_distortion,
                                                        right_image,
                                                        right_intrinsics,
                                                        right_distortion,
                                                        l2r_rmat,
                                                        l2r_tvec
                                                        )
    camera_to_model = np.linalg.inv(registration)

    print(f'Input model: {full_pointcloud}')
    print(f'{len(pointcloud)} points in point cloud')
    print(residual)
    print(registration)

    reproject_and_save(left_image, camera_to_model, pointcloud, left_intrinsics,
                       outfile='tests/data/synthetic_liver/output_full_pointcloud.png')

    # Run pipeline using reduced liver model, where most of the non visible
    # points have been deleted. Should give a better result.
    pointcloud = np.loadtxt(front_surface_only_pointcloud)
    residual1, registration = reg_points_to_vid.register(pointcloud,
                                                        left_image,
                                                        left_intrinsics,
                                                        left_distortion,
                                                        right_image,
                                                        right_intrinsics,
                                                        right_distortion,
                                                        l2r_rmat,
                                                        l2r_tvec,

                                                        )
    camera_to_model = np.linalg.inv(registration)

    print(f'Input model: {front_surface_only_pointcloud}')
    print(f'{len(pointcloud)} points in point cloud')
    print(residual)
    print(registration)

    # Reset input image, to remove the pixels from the first projection
    left_image = cv2.imread('tests/data/synthetic_liver/synthetic-left.png')
    reproject_and_save(left_image, camera_to_model, pointcloud, left_intrinsics,
                       outfile='tests/data/synthetic_liver/output_reduced_pointcloud.png')



