# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for rigid registration. """

import numpy as np


class RigidRegistration:

    def register(self,
                 fixed_cloud: np.ndarray,
                 moving_cloud: np.ndarray
                 ):
        """
        A derived class must implement this.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Nx3] moving point cloud.
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """
        raise NotImplementedError("Derived classes should implement this.")
