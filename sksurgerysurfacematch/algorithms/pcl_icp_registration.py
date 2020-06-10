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
                 fixed_cloud: np.ndarray,
                 moving_cloud: np.ndarray
                 ):
        """
        Uses PCL library, wrapped in scikit-surgerypclcpp.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Mx3] moving point cloud.
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """
        transform = np.eye(4)
        residual = pclp.iterative_closest_point(moving_cloud,  # source
                                                fixed_cloud,   # target
                                                transform)     # 4x4
        return residual, transform
