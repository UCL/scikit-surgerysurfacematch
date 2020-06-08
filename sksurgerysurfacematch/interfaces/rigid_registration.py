# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for rigid registration. """

import numpy as np


class RigidRegistration:
    """
    Base class for classes that can rigidly register (align), two point clouds.
    """
    def register(self,
                 fixed_cloud: np.ndarray,
                 moving_cloud: np.ndarray
                 ):
        """
        A derived class must implement this.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Mx3] moving point cloud.
        :return: residual, [4x4] transformation matrix, moving-to-fixed space.
        """
        raise NotImplementedError("Derived classes should implement this.")
