# -*- coding: utf-8 -*-

""" Various utilities, mainly to help testing. """

import copy
import cv2


def reproject_and_save(image,
                       model_to_camera,
                       point_cloud,
                       camera_matrix,
                       output_file):
    """
    For testing purposes, projects points onto image, and writes to file.

    :param image: BGR image, undistorted.
    :param model_to_camera: [4x4] ndarray of model-to-camera transform
    :param point_cloud: [Nx3] ndarray of cloud of points to project
    :param camera_matrix: [3x3] OpenCV camera_matrix (intrinsics)
    :param output_file: file name
    """
    output_image = copy.deepcopy(image)

    rmat = model_to_camera[:3, :3]
    rvec = cv2.Rodrigues(rmat)[0]
    tvec = model_to_camera[:3, 3]

    projected, _ = cv2.projectPoints(point_cloud,
                                     rvec,
                                     tvec,
                                     camera_matrix,
                                     None)

    for i in range(projected.shape[0]):

        x_c, y_c = projected[i][0]
        x_c = int(x_c)
        y_c = int(y_c)

        # Skip points that aren't in the bounds of image
        if x_c <= 0 or x_c >= output_image.shape[1]:
            continue
        if y_c <= 0 or y_c >= output_image.shape[0]:
            continue

        output_image[y_c, x_c, :] = [255, 0, 0]

    cv2.imwrite(output_file, output_image)
