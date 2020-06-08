# -*- coding: utf-8 -*-

""" Pipeline to register 3D point cloud to 2D stereo video """

import numpy as np
import sksurgerypclpython as pclp
import sksurgerysurfacematch.interfaces.video_segmentor as vs
import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr
import sksurgerysurfacematch.interfaces.rigid_registration as rr


class Register3DToStereoVideo:
    """
    Constructor, uses Dependency Injection for each main functionality.
    """
    def __init__(self,
                 video_segmentor: vs.VideoSegmentor,
                 surface_reconstructor: sr.StereoReconstructor,
                 rigid_registration: rr.RigidRegistration,
                 left_mask: np.ndarray = None,
                 right_mask: np.ndarray = None,
                 voxel_reduction: list = None,
                 statistical_outlier_reduction: list = None
                 ):
        self.video_segmentor = video_segmentor
        self.surface_reconstructor = surface_reconstructor
        self.rigid_registration = rigid_registration
        self.left_static_mask = left_mask
        self.right_static_mask = right_mask
        self.voxel_reduction = voxel_reduction
        self.statistical_outlier_reduction = statistical_outlier_reduction

    def register(self,
                 point_cloud: np.ndarray,
                 left_image: np.ndarray,
                 left_camera_matrix: np.ndarray,
                 left_dist_coeffs: np.ndarray,
                 right_image: np.ndarray,
                 right_camera_matrix: np.ndarray,
                 right_dist_coeffs: np.ndarray,
                 left_to_right_rmat: np.ndarray,
                 left_to_right_tvec: np.ndarray
                 ):
        """
        Main method to do a single 3D cloud to 2D stereo video registration.

        Camera calibration parameters are in OpenCV format.

        :param point_cloud: [Nx3] points, each row, x,y,z, e.g. from CT/MR.
        :param left_image: undistorted, BGR image
        :param left_camera_matrix: [3x3] camera matrix.
        :param left_dist_coeffs: [1x5] distortion coeff's.
        :param right_image: undistorted, BGR image
        :param right_camera_matrix: [3x3] camera matrix.
        :param right_dist_coeffs: [1x5] distortion coeff's.
        :param left_to_right_rmat: [3x3] left-to-right rotation matrix.
        :param left_to_right_tvec: [1x3] left-to-right translation vector.
        :return: [4x4] matrix, of point_cloud to left camera space.
        """
        left_mask = np.ones((left_image.shape[0],
                             left_image.shape[1])) * 255
        right_mask = np.ones((right_image.shape[0],
                             left_image.shape[1])) * 255

        if self.video_segmentor is not None:
            left_mask = self.video_segmentor.segment(left_image)
            right_mask = self.video_segmentor.segment(right_image)

        if self.left_static_mask is not None:
            left_mask = np.logical_and(left_mask, self.left_static_mask)

        if self.right_static_mask is not None:
            right_mask = np.logical_and(right_mask, self.right_static_mask)

        reconstruction = \
            self.surface_reconstructor.reconstruct(left_image,
                                                   left_camera_matrix,
                                                   left_dist_coeffs,
                                                   right_image,
                                                   right_camera_matrix,
                                                   right_dist_coeffs,
                                                   left_to_right_rmat,
                                                   left_to_right_tvec,
                                                   left_mask,
                                                   right_mask
                                                   )

        reconstruction = reconstruction[:, 0:3]

        if self.voxel_reduction is not None:
            reconstruction = \
                pclp.down_sample_points(
                    reconstruction,
                    self.voxel_reduction[0],
                    self.voxel_reduction[1],
                    self.voxel_reduction[2])

        if self.statistical_outlier_reduction is not None:
            reconstruction = \
                pclp.remove_outlier_points(
                    reconstruction,
                    self.statistical_outlier_reduction[0],
                    self.statistical_outlier_reduction[1])

        # We register the fixed point cloud to the reconstructed point cloud.
        registration = self.rigid_registration.register(point_cloud,
                                                        reconstruction
                                                        )

        # .. and then invert the result.
        registration = np.linalg.inv(registration)

        return registration
