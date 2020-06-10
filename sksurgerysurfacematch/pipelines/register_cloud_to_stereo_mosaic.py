# -*- coding: utf-8 -*-

""" Pipeline to register 3D point cloud to mosaiced surface reconstruction.  """

import copy
import numpy as np
import cv2
import sksurgerysurfacematch.interfaces.video_segmentor as vs
import sksurgerysurfacematch.interfaces.stereo_reconstructor as sr
import sksurgerysurfacematch.interfaces.rigid_registration as rr


# pylint:disable=too-many-instance-attributes
class Register3DToMosaicedStereoVideo:
    """
    Class to register a point cloud to a series of surfaces
    derived from stereo video, and stitched together.
    """
    def __init__(self,
                 video_segmentor: vs.VideoSegmentor,
                 surface_reconstructor: sr.StereoReconstructor,
                 rigid_registration: rr.RigidRegistration,
                 number_of_keypoints: int = 25,
                 left_mask: np.ndarray = None,
                 voxel_reduction: list = None,
                 statistical_outlier_reduction: list = None
                 ):
        """
        Uses Dependency Injection for each main component.

        :param video_segmentor: Optional class to pre-segment the video.
        :param surface_reconstructor: Mandatory class to do reconstruction.
        :param rigid_registration: Mandatory class to perform rigid alignment.
        :param number_of_keypoints: Number of keypoints to use for matching.
        :param left_mask: a static mask to apply to stereo reconstruction.
        :param voxel_reduction: [vx, vy, vz] parameters for PCL
        Voxel Grid reduction.
        :param statistical_outlier_reduction: [meanK, StdDev] parameters for
        PCL Statistical Outlier Removal filter.
        """
        self.video_segmentor = video_segmentor
        self.surface_reconstructor = surface_reconstructor
        self.rigid_registration = rigid_registration
        self.number_of_keypoints = number_of_keypoints
        self.left_static_mask = left_mask
        self.voxel_reduction = voxel_reduction
        self.statistical_outlier_reduction = statistical_outlier_reduction
        self.previous_image = None
        self.previous_recon = None
        self.previous_keypoints = None
        self.previous_descriptors = None

    def reset(self):
        """
        Reset's internal data members, so that you can start accumulating
        data again.
        """
        self.previous_image = None
        self.previous_recon = None
        self.previous_keypoints = None
        self.previous_descriptors = None

    def grab(self,
             left_image: np.ndarray,
             left_camera_matrix: np.ndarray,
             left_dist_coeffs: np.ndarray,
             right_image: np.ndarray,
             right_camera_matrix: np.ndarray,
             right_dist_coeffs: np.ndarray,
             left_to_right_rmat: np.ndarray,
             left_to_right_tvec: np.ndarray):
        """
        To do, explain.

        :param left_image:
        :param left_camera_matrix:
        :param left_dist_coeffs:
        :param right_image:
        :param right_camera_matrix:
        :param right_dist_coeffs:
        :param left_to_right_rmat:
        :param left_to_right_tvec:
        :return:
        """
        left_mask = np.ones((left_image.shape[0],
                             left_image.shape[1])) * 255

        if self.video_segmentor is not None:
            left_mask = self.video_segmentor.segment(left_image)

        if self.left_static_mask is not None:
            left_mask = np.logical_and(left_mask, self.left_static_mask)

        orb = cv2.ORB_create()

        current_key_points, current_descriptors = \
            orb.detectAndCompute(left_image, None)

        if self.previous_image is None:
            full_reconstruction = \
                self.surface_reconstructor.reconstruct(left_image,
                                                       left_camera_matrix,
                                                       left_dist_coeffs,
                                                       right_image,
                                                       right_camera_matrix,
                                                       right_dist_coeffs,
                                                       left_to_right_rmat,
                                                       left_to_right_tvec,
                                                       left_mask
                                                       )
            self.previous_image = copy.deepcopy(left_image)
            self.previous_recon = full_reconstruction
            self.previous_keypoints = current_key_points
            self.previous_descriptors = current_descriptors


        # Match
        b_f = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = b_f.match(self.previous_descriptors, current_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = matches[:self.number_of_keypoints]
        print(best_matches)
