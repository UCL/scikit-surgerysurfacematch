# -*- coding: utf-8 -*-

""" Various registration routines to reduce duplication. """

import numpy as np
import sksurgerycore.transforms.matrix as mt
import sksurgerysurfacematch.interfaces.rigid_registration as rr


def do_rigid_registration(reconstructed_cloud,
                          reference_cloud,
                          rigid_registration: rr.RigidRegistration,
                          initial_transform: np.ndarray = None,
                          ):
    """
    Triggers a rigid body registration using rigid_registration.
    :param reconstructed_cloud: [Nx3] point cloud, e.g. from video.
    :param reference_cloud: [Mx3] point cloud, e.g. from CT/MR
    :param rigid_registration: Object that implements a rigid registration.
    :param initial_transform: [4x4] ndarray representing an initial estimate.
    :return: residual (float), [4x4] transform
    """
    point_cloud = reconstructed_cloud

    if initial_transform is not None:
        point_cloud = \
            np.matmul(
                initial_transform[0:3, 0:3], np.transpose(reference_cloud)) \
            + initial_transform[0:3, 3].reshape((3, 1))
        point_cloud = np.transpose(point_cloud)

    # Do registration. Best to register recon points to
    # the provided model (likely from CT or MR), and then invert.
    residual, transform = \
        rigid_registration.register(reconstructed_cloud,
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
