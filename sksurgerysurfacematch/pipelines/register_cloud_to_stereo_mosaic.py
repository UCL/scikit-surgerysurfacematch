# -*- coding: utf-8 -*-

""" Pipeline to register 3D point cloud to mosaic'ed surface reconstruction. """

import copy
import numpy as np
import cv2
import sksurgerypclpython as pclp
import sksurgeryopencvpython as cvcpp
import sksurgerycore.algorithms.procrustes as proc
import sksurgerysurfacematch.utils.registration_utils as ru
import sksurgerysurfacematch.interfaces.video_segmentor as vs
import sksurgerysurfacematch.algorithms.reconstructor_with_rectified_images \
    as sr
import sksurgerysurfacematch.interfaces.rigid_registration as rr


# pylint:disable=too-many-instance-attributes, invalid-name, too-many-locals
# pylint:disable=unsubscriptable-object, too-many-arguments,too-many-branches


class Register3DToMosaicedStereoVideo:
    """
    Class to register a point cloud to a series of surfaces
    derived from stereo video, and stitched together.

    Uses Dependency Injection for each pluggable component.

    :param video_segmentor: Optional class to pre-segment the video.
    :param surface_reconstructor: Mandatory class to do reconstruction.
    :param rigid_registration: Mandatory class to perform rigid alignment.
    :param left_camera_matrix: [3x3] camera matrix.
    :param right_camera_matrix: [3x3] camera matrix.
    :param left_to_right_rmat: [3x3] left-to-right rotation matrix.
    :param left_to_right_tvec: [1x3] left-to-right translation vector.
    :param min_number_of_keypoints: Number of keypoints to use for matching.
    :param max_fre_threshold: maximum FRE when stitching frames together.
    :param left_mask: a static mask to apply to stereo reconstruction.
    :param z_range: [min range, max range] to limit reconstructed points.
    :param radius_removal: [radius, number] to reject points with too few \
    neighbours
    :param voxel_reduction: [vx, vy, vz] parameters for PCL
    Voxel Grid reduction.
    """
    def __init__(
            self,
            video_segmentor: vs.VideoSegmentor,
            surface_reconstructor: sr.StereoReconstructorWithRectifiedImages,
            rigid_registration: rr.RigidRegistration,
            left_camera_matrix: np.ndarray,
            right_camera_matrix: np.ndarray,
            left_to_right_rmat: np.ndarray,
            left_to_right_tvec: np.ndarray,
            min_number_of_keypoints: int = 25,
            max_fre_threshold=2,
            left_mask: np.ndarray = None,
            z_range: list = None,
            radius_removal: list = None,
            voxel_reduction: list = None):

        self.video_segmentor = video_segmentor
        self.surface_reconstructor = surface_reconstructor
        self.rigid_registration = rigid_registration
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.left_to_right_rmat = left_to_right_rmat
        self.left_to_right_tvec = left_to_right_tvec
        self.min_number_of_keypoints = min_number_of_keypoints
        self.max_fre_threshold = max_fre_threshold
        self.left_static_mask = left_mask
        self.z_range = z_range
        self.radius_removal = radius_removal
        self.voxel_reduction = voxel_reduction
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
             right_image: np.ndarray):
        """
        Call this repeatedly to grab a surface and use ORM key points to
        match previous reconstruction to the current frame.

        :param left_image: undistorted, BGR image
        :param right_image: undistorted, BGR image
        """
        left_mask = np.ones((left_image.shape[0],
                             left_image.shape[1]), dtype=np.uint8) * 255

        if self.video_segmentor is not None:
            left_mask = self.video_segmentor.segment(left_image)
            left_mask = 255 * (left_mask > 0)

        if self.left_static_mask is not None:
            left_mask = np.bitwise_and(left_mask, self.left_static_mask)

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
                                                       self.left_camera_matrix,
                                                       self.right_camera_matrix,
                                                       self.left_to_right_rmat,
                                                       self.left_to_right_tvec)

            full_reconstruction = \
                self.surface_reconstructor.reconstruct(left_image,
                                                       self.left_camera_matrix,
                                                       right_image,
                                                       self.right_camera_matrix,
                                                       self.left_to_right_rmat,
                                                       self.left_to_right_tvec,
                                                       left_mask
                                                       )

            full_reconstruction = full_reconstruction[:, 0:3]

            if self.z_range is not None:
                full_reconstruction = \
                    pclp.pass_through_filter(full_reconstruction,
                                             'z',
                                             self.z_range[0],
                                             self.z_range[1],
                                             True)

            if self.radius_removal is not None:
                full_reconstruction = \
                    pclp.radius_removal_filter(full_reconstruction,
                                               self.radius_removal[0],
                                               self.radius_removal[1])

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

                    if fre < self.max_fre_threshold:
                        # Transform previous point cloud to current
                        transformed_point_cloud = \
                            np.transpose(np.matmul(rmat,
                                                   np.transpose(
                                                       self.previous_recon))
                                         + tvec)

                        # Combine and simplify? or just combine?
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

    def register(self,
                 point_cloud: np.ndarray,
                 initial_transform: np.ndarray = None
                 ):
        """
        Registers a point cloud to the internal mosaicc'ed reconstruction.

        :param point_cloud: [Nx3] points, each row, x,y,z, e.g. from CT/MR.
        :param initial_transform: [4x4] of initial rigid transform.
        :return: residual, [4x4] transform, of point_cloud to left camera space,
        and [Mx6] reconstructed point cloud, as [x, y, z, r, g, b] rows.
        """
        if self.previous_recon is None:
            raise ValueError("No reconstruction has been performed")

        recon_points = self.previous_recon

        if self.voxel_reduction is not None:
            recon_points = \
                pclp.down_sample_points(
                    recon_points,
                    self.voxel_reduction[0],
                    self.voxel_reduction[1],
                    self.voxel_reduction[2])

        residual, transform = ru.do_rigid_registration(recon_points,
                                                       point_cloud,
                                                       self.rigid_registration,
                                                       initial_transform)

        # Don't return a pointer to internal self.previous_recon.
        return residual, transform, copy.deepcopy(recon_points)
