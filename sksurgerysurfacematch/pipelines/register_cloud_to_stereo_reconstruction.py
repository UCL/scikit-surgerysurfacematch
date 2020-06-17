# -*- coding: utf-8 -*-

""" Pipeline to register 3D point cloud to 2D stereo video """

import numpy as np
import sksurgerycore.transforms.matrix as mt
import sksurgerypclpython as pclp
import sksurgerysurfacematch.interfaces.video_segmentor as vs
import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr
import sksurgerysurfacematch.interfaces.rigid_registration as rr


class Register3DToStereoVideo:
    """
    Class for single-shot, registration of 3D point cloud to stereo video.
    """
    def __init__(self,
                 video_segmentor: vs.VideoSegmentor,
                 surface_reconstructor: sr.StereoReconstructor,
                 rigid_registration: rr.RigidRegistration,
                 left_camera_matrix: np.ndarray,
                 left_dist_coeffs: np.ndarray,
                 right_camera_matrix: np.ndarray,
                 right_dist_coeffs: np.ndarray,
                 left_to_right_rmat: np.ndarray,
                 left_to_right_tvec: np.ndarray,
                 left_mask: np.ndarray = None,
                 z_range: list = None,
                 voxel_reduction: list = None,
                 statistical_outlier_reduction: list = None
                 ):
        """
        Uses Dependency Injection for each main component.

        :param video_segmentor: Optional class to pre-segment the video.
        :param surface_reconstructor: Mandatory class to do reconstruction.
        :param rigid_registration: Mandatory class to perform rigid alignment.
        :param left_camera_matrix: [3x3] camera matrix.
        :param left_dist_coeffs: [1x5] distortion coeff's.
        :param right_camera_matrix: [3x3] camera matrix.
        :param right_dist_coeffs: [1x5] distortion coeff's.
        :param left_to_right_rmat: [3x3] left-to-right rotation matrix.
        :param left_to_right_tvec: [1x3] left-to-right translation vector.
        :param left_mask: a static mask to apply to stereo reconstruction.
        :param z_range: [min range, max range] to limit reconstructed points
        :param voxel_reduction: [vx, vy, vz] parameters for PCL
        Voxel Grid reduction.
        :param statistical_outlier_reduction: [meanK, StdDev] parameters for
        PCL Statistical Outlier Removal filter.
        """
        self.video_segmentor = video_segmentor
        self.surface_reconstructor = surface_reconstructor
        self.rigid_registration = rigid_registration
        self.left_camera_matrix = left_camera_matrix
        self.left_dist_coeffs = left_dist_coeffs
        self.right_camera_matrix = right_camera_matrix
        self.right_dist_coeffs = right_dist_coeffs
        self.left_to_right_rmat = left_to_right_rmat
        self.left_to_right_tvec = left_to_right_tvec
        self.left_static_mask = left_mask
        self.z_range = z_range
        self.voxel_reduction = voxel_reduction
        self.statistical_outlier_reduction = statistical_outlier_reduction

    # pylint: disable=too-many-arguments
    def register(self,
                 point_cloud: np.ndarray,
                 left_image: np.ndarray,
                 right_image: np.ndarray,
                 initial_transform: np.ndarray = None
                 ):
        """
        Main method to do a single 3D cloud to 2D stereo video registration.

        Camera calibration parameters are in OpenCV format.

        :param point_cloud: [Nx3] points, each row, x,y,z, e.g. from CT/MR.
        :param left_image: undistorted, BGR image
        :param right_image: undistorted, BGR image
        :param initial_transform: [4x4] of initial rigid transform.
        :return: residual, [4x4] matrix, of point_cloud to left camera space.
        """
        left_mask = None

        if self.left_static_mask is not None:
            left_mask = self.left_static_mask

        if self.video_segmentor is not None:
            dynamic_mask = self.video_segmentor.segment(left_image)

            if left_mask is None:
                left_mask = dynamic_mask
            else:
                left_mask = np.logical_and(left_mask, dynamic_mask)

        if left_mask is None:
            left_mask = np.ones((left_image.shape[0],
                                 left_image.shape[1])) * 255

        reconstruction = \
            self.surface_reconstructor.reconstruct(left_image,
                                                   self.left_camera_matrix,
                                                   self.left_dist_coeffs,
                                                   right_image,
                                                   self.right_camera_matrix,
                                                   self.right_dist_coeffs,
                                                   self.left_to_right_rmat,
                                                   self.left_to_right_tvec,
                                                   left_mask
                                                   )

        recon_points = reconstruction[:, 0:3]

        if self.z_range is not None:
            recon_points = pclp.pass_through_filter(recon_points,
                                                    'z',
                                                    self.z_range[0],
                                                    self.z_range[1],
                                                    True)

        if self.voxel_reduction is not None:
            recon_points = \
                pclp.down_sample_points(
                    recon_points,
                    self.voxel_reduction[0],
                    self.voxel_reduction[1],
                    self.voxel_reduction[2])

        if self.statistical_outlier_reduction is not None:
            recon_points = \
                pclp.remove_outlier_points(
                    recon_points,
                    self.statistical_outlier_reduction[0],
                    self.statistical_outlier_reduction[1])

        # Do initial alignment if we have one.
        if initial_transform is not None:
            point_cloud = \
                np.matmul(
                    initial_transform[0:3, 0:3], np.transpose(point_cloud)) \
                + initial_transform[0:3, 3].reshape((3, 1))
            point_cloud = np.transpose(point_cloud)

        # Do registration. Best to register recon points to
        # the provided model (likely from CT or MR), and then invert.
        residual, transform = \
            self.rigid_registration.register(recon_points,
                                             point_cloud
                                             )
        transform = np.linalg.inv(transform)

        # Combine initial, if we have one.
        if initial_transform is not None:
            init_mat = \
                mt.construct_rigid_transformation(
                    initial_transform[0:3, 0:3],
                    initial_transform[0:3, 3]
                )
            transform = np.matmul(transform, init_mat)

        return residual, transform
