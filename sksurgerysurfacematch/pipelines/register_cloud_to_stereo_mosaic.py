# -*- coding: utf-8 -*-

""" Pipeline to register 3D point cloud to mosaic'ed surface reconstruction. """

import numpy as np
import cv2
import sksurgeryopencvpython as cvcpp
import sksurgerycore.algorithms.procrustes as proc
import sksurgerysurfacematch.interfaces.video_segmentor as vs
import sksurgerysurfacematch.algorithms.reconstructor_with_rectified_images \
    as sr
import sksurgerysurfacematch.interfaces.rigid_registration as rr


# pylint:disable=too-many-instance-attributes, invalid-name, too-many-locals
# pylint:disable=unsubscriptable-object

class Register3DToMosaicedStereoVideo:
    """
    Class to register a point cloud to a series of surfaces
    derived from stereo video, and stitched together.
    """
    def __init__(
            self,
            video_segmentor: vs.VideoSegmentor,
            surface_reconstructor: sr.StereoReconstructorWithRectifiedImages,
            rigid_registration: rr.RigidRegistration,
            min_number_of_keypoints: int = 25,
            left_mask: np.ndarray = None,
            voxel_reduction: list = None,
            statistical_outlier_reduction: list = None):
        """
        Uses Dependency Injection for each main component.

        :param video_segmentor: Optional class to pre-segment the video.
        :param surface_reconstructor: Mandatory class to do reconstruction.
        :param rigid_registration: Mandatory class to perform rigid alignment.
        :param min_number_of_keypoints: Number of keypoints to use for matching.
        :param left_mask: a static mask to apply to stereo reconstruction.
        :param voxel_reduction: [vx, vy, vz] parameters for PCL
        Voxel Grid reduction.
        :param statistical_outlier_reduction: [meanK, StdDev] parameters for
        PCL Statistical Outlier Removal filter.
        """
        self.video_segmentor = video_segmentor
        self.surface_reconstructor = surface_reconstructor
        self.rigid_registration = rigid_registration
        self.min_number_of_keypoints = min_number_of_keypoints
        self.left_static_mask = left_mask
        self.voxel_reduction = voxel_reduction
        self.statistical_outlier_reduction = statistical_outlier_reduction
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.previous_good_l2r_matches = None
        self.previous_good_l2r_matched_descriptors = None
        self.previous_triangulated_key_points = None
        self.previous_recon = None

    def reset(self):
        """
        Reset's internal data members, so that you can start accumulating
        data again.
        """
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.previous_good_l2r_matches = None
        self.previous_good_l2r_matched_descriptors = None
        self.previous_triangulated_key_points = None
        self.previous_recon = None

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

        current_left_key_points, current_left_descriptors = \
            orb.detectAndCompute(left_image, None)

        current_right_key_points, current_right_descriptors = \
            orb.detectAndCompute(right_image, None)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)  # 6=FLANN_INDEX_LSH
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        left_to_right_matches = flann.knnMatch(
            current_left_descriptors,
            current_right_descriptors, k=2)

        # Keep good matches, based on Lowe's ratio test.
        good_l2r_matches = []
        for m, n in left_to_right_matches:
            if m.distance < 0.7 * n.distance:
                good_l2r_matches.append(m)

        if len(good_l2r_matches) > self.min_number_of_keypoints:

            left_descriptors = np.zeros((len(good_l2r_matches),
                                         len(current_left_descriptors[0])),
                                        dtype=np.uint8)
            for i, m in enumerate(good_l2r_matches):
                left_descriptors[i, :] = current_left_descriptors[m.queryIdx]

            left_pts = np.float32([current_left_key_points[m.queryIdx].pt
                                   for m in good_l2r_matches])
            right_pts = np.float32([current_right_key_points[m.trainIdx].pt
                                    for m in good_l2r_matches])
            paired_pts = np.zeros((left_pts.shape[0], 4))
            paired_pts[:, 0:2] = left_pts
            paired_pts[:, 2:4] = right_pts
            triangulated_l2r_key_points = \
                cvcpp.triangulate_points_using_hartley(paired_pts,
                                                       left_camera_matrix,
                                                       right_camera_matrix,
                                                       left_to_right_rmat,
                                                       left_to_right_tvec)

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
            full_reconstruction = full_reconstruction[:, 0:3]

            # Match to previous frame
            if self.previous_keypoints is not None and \
                    self.previous_descriptors is not None:

                previous_to_current = flann.knnMatch(
                    self.previous_good_l2r_matched_descriptors,
                    left_descriptors,
                    k=2)

                # Keep good matches, based on Lowe's ratio test.
                good_prev_to_current_matches = []
                for i, id_pair in enumerate(previous_to_current):
                    if id_pair is not None and len(id_pair) == 2:
                        m, n = id_pair
                        if m.distance < 0.7 * n.distance:
                            good_prev_to_current_matches.append(m)

                if len(good_prev_to_current_matches) > \
                        self.min_number_of_keypoints:

                    prev_3d_pts = np.float32(
                        [self.previous_triangulated_key_points[m.queryIdx]
                         for m in good_prev_to_current_matches])
                    current_3d_pts = np.float32(
                        [triangulated_l2r_key_points[m.trainIdx]
                         for m in good_prev_to_current_matches])

                    # Compute rigid body transform. Maybe use RANSAC?
                    rmat, tvec, fre = proc.orthogonal_procrustes(current_3d_pts,
                                                                 prev_3d_pts)

                    if fre < 1:
                        # Transform previous point cloud to current
                        transformed_point_cloud = \
                            np.transpose(np.matmul(rmat,
                                                   np.transpose(
                                                       self.previous_recon))
                                         + tvec)

                        # Combine and simplify?
                        full_reconstruction = \
                            np.vstack((transformed_point_cloud,
                                       full_reconstruction))

            # Save current iteration, such that next iteration it will
            # be called the 'previous' iteration, for tracking purposes.
            self.previous_keypoints = current_left_key_points
            self.previous_descriptors = current_left_descriptors
            self.previous_good_l2r_matches = good_l2r_matches
            self.previous_good_l2r_matched_descriptors = left_descriptors
            self.previous_triangulated_key_points = triangulated_l2r_key_points
            self.previous_recon = full_reconstruction
