
import cv2
import numpy as np

import sksurgeryvtk.models.vtk_surface_model as sksvtk
import sksurgerysurfacematch.algorithms.sgbm_reconstructor as sr
import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.pipelines.register_cloud_to_stereo_reconstruction as reg


def reproject_and_save(image,
                       model_to_camera,
                       pointcloud, 
                       intrinsics, 
                       outfile):

    """ Project model points back to 2D and save image. """
    rmat = model_to_camera[:3, :3]
    rvec = cv2.Rodrigues(rmat)[0]
    tvec = model_to_camera[:3, 3]

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

    left_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.intrinsics.txt')
    right_intrinsics = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.left.distortion.txt')
    right_distortion = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.right.distortion.txt')
    l2r_matrix = np.loadtxt('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/calib.l2r.txt')

    l2r_rmat = l2r_matrix[0:3, 0:3]
    l2r_tvec = l2r_matrix[0:3, 3:4]

    pointcloud_file = 'tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_CT_cut.stl'
    model = sksvtk.VTKSurfaceModel(pointcloud_file, (1.0, 1.0, 1.0))

    left_image = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_left.bmp')
    left_undistorted = cv2.undistort(left_image, left_intrinsics, left_distortion)
    left_mask = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_MASK_eval_inverted.png')
    left_mask = cv2.cvtColor(left_mask, cv2.COLOR_BGR2GRAY)
    right_image = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_right.bmp')
    right_undistorted = cv2.undistort(right_image, right_intrinsics, right_distortion)

    point_cloud = model.get_points_as_numpy()
    model_to_camera = np.eye(4)

    # Produce picture of gold standard registration.
    reproject_and_save(left_undistorted, model_to_camera, point_cloud, left_intrinsics,
                       outfile='tests/output/open_cas_tmi_gold.png')

    # Create registration pipeline.
    reg_points_to_vid = \
        reg.Register3DToStereoVideo(None,
                                    sr.SGBMReconstructor(),
                                    pir.RigidRegistration(),
                                    left_mask=left_mask,
                                    z_range=[45, 65],
                                    voxel_reduction=[5, 5, 5],
                                    statistical_outlier_reduction=[10, 3]
                                    )

    residual, registration = reg_points_to_vid.register(point_cloud,
                                                        left_undistorted,
                                                        left_intrinsics,
                                                        left_distortion,
                                                        right_undistorted,
                                                        right_intrinsics,
                                                        right_distortion,
                                                        l2r_rmat,
                                                        l2r_tvec,
                                                        model_to_camera
                                                        )

    print(f'Model: {pointcloud_file}')
    print(f'{len(point_cloud)} points in reference point cloud')

    print("Residual:" + str(residual))
    print("Registration, full point cloud:\n" + str(registration))

    left_image = cv2.imread('tests/data/open_cas_tmi/Stereo_SD_d_complete_22/Stereo_SD_d_complete_22_IMG_left.bmp')
    left_undistorted = cv2.undistort(left_image, left_intrinsics, left_distortion)
    reproject_and_save(left_undistorted, registration, point_cloud, left_intrinsics,
                       outfile='tests/output/open_cas_tmi_registered.png')

    assert residual < 5.0

