# -*- coding: utf-8 -*-

""" PCL ICP implementation of RigidRegistration interface. """

import copy
import numpy as np
import sksurgerypclpython as pclp
import sksurgerysurfacematch.interfaces.rigid_registration as rr


class RigidRegistration(rr.RigidRegistration):
    """
    Class that uses PCL implementation of ICP to register fixed/moving clouds.
    """

    def __init__(self,
                 max_iterations: int = 100,
                 max_correspondence_threshold: float = 1,
                 transformation_epsilon: float = 0.0001,
                 fitness_epsilon: float = 0.0001,
                 use_lm_icp: bool = True,
                 ):
        """
        :param max_iterations: maximum number of ICP iterations, defaults to 100
        :type max_iterations: int, optional
        :param max_correspondence_threshold: reject pairs if distance > thresh,\
             defaults to 1
        :type max_correspondence_threshold: float, optional
        :param transformation_epsilon: early exit based on transformation \
            params, defaults to 0.0001
        :type transformation_epsilon: float, optional
        :param fitness_epsilon: early exit based on cost function, \
             defaults to 0.0001
        :type fitness_epsilon: float, optional
        :param use_lm_icp: Use LM-ICP if true, otherwise normal ICP, \
            defaults to True
        :type use_lm_icp: bool, optional
        """

        self.max_iterations = max_iterations
        self.max_correspondence_threshold = max_correspondence_threshold
        self.transformation_epsilon = transformation_epsilon
        self.fitness_epsilon = fitness_epsilon
        self.use_lm_icp = use_lm_icp

    def register(self,
                 moving_cloud: np.ndarray,
                 fixed_cloud: np.ndarray,
                 ):
        """
        Uses PCL library, wrapped in scikit-surgerypclcpp.

        :param moving_cloud: [Nx3] source/moving point cloud.
        :param fixed_cloud: [Mx3] target/fixed point cloud.

        :return: [4x4] transformation matrix, moving-to-fixed space.
        """
        transformed_source = copy.deepcopy(moving_cloud)

        transform = np.eye(4)

        residual = pclp.icp(moving_cloud,
                            fixed_cloud,
                            self.max_iterations,
                            self.max_correspondence_threshold,
                            self.transformation_epsilon,
                            self.fitness_epsilon,
                            self.use_lm_icp,
                            transform,
                            transformed_source)

        return residual, transform
