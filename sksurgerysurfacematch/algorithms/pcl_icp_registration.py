# -*- coding: utf-8 -*-

""" PCL ICP implementation of RigidRegistration interface. """

import numpy as np
import sksurgerypclpython as pclp
import sksurgerysurfacematch.interfaces.rigid_registration as rr


class RigidRegistration(rr.RigidRegistration):
    """
    Class that uses PCL implementation of ICP to register fixed/moving clouds.
    At the moment, we are just relying on all default parameters.
    """
    def register(self,
                 source_cloud: np.ndarray,
                 target_cloud: np.ndarray
                 ):
        """
        Uses PCL library, wrapped in scikit-surgerypclcpp.

        :param source_cloud: [Nx3] source/moving point cloud.
        :param target_cloud: [Mx3] target/fixed point cloud.
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """
        transform = np.eye(4)
        residual = pclp.iterative_closest_point(source_cloud,  # source
                                                target_cloud,  # target
                                                transform)     # 4x4
        return residual, transform
