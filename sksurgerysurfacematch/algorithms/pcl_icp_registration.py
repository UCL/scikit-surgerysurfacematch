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
    def register(self,
                 source_cloud: np.ndarray,
                 target_cloud: np.ndarray,
                 max_iterations=100,
                 max_correspondence_threshold=1,
                 transformation_epsilon=0.0001,
                 fitness_epsilon=0.0001
                 ):
        """
        Uses PCL library, wrapped in scikit-surgerypclcpp.

        :param source_cloud: [Nx3] source/moving point cloud.
        :param target_cloud: [Mx3] target/fixed point cloud.
        :param normal_search_radius: radius to search, for surface normals.
        :param normal_tolerance_in_degrees: reject pairs if normals > thresh
        :param max_iterations: maximum number of ICP iterations
        :param max_correspondence_threshold: reject pairs if distance > thresh
        :param transformation_epsilon: early exit based on transformation params
        :param fitness_epsilon: early exit based on cost function
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """
        transformed_source = copy.deepcopy(source_cloud)

        transform = np.eye(4)

        residual = pclp.icp(source_cloud,
                            target_cloud,
                            max_iterations,
                            max_correspondence_threshold,
                            transformation_epsilon,
                            fitness_epsilon,
                            True,
                            transform,
                            transformed_source)

        return residual, transform
