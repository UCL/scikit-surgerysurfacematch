# -*- coding: utf-8 -*-

""" Go ICP implementation of RigidRegistration interface. """

# pylint:disable=invalid-name, no-name-in-module

import logging
import numpy as np
from sksurgerygoicppython import  GoICP, POINT3D, ROTNODE, TRANSNODE
import sksurgerysurfacematch.interfaces.rigid_registration as rr

LOGGER = logging.getLogger(__name__)

def numpy_to_POINT3D_array(numpy_pointcloud):
    """ Covert numpy array to POINT3D array suitable for GoICP algorithm."""
    plist = numpy_pointcloud.tolist()
    p3dlist = []
    for x, y, z in plist:
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return numpy_pointcloud.shape[0], p3dlist


def create_scaling_matrix(scale: float) -> np.ndarray:
    """ Create a scaling matrix, with the same value in each axis. """
    matrix = np.eye(4)

    matrix[0][0] = scale
    matrix[1][1] = scale
    matrix[2][2] = scale

    return matrix

def create_translation_matrix(translate: np.ndarray) -> np.ndarray:
    """ Create translation matrix from 3x1 translation vector. """
    matrix = np.eye(4)
    matrix[:3, 3] = translate

    return matrix

def demean_and_normalise(points_a: np.ndarray,
                         points_b: np.ndarray):
    """
    Independently centre each point cloud around 0,0,0, then normalise
    both to [-1,1].

    :param points_a: 1st point cloud
    :type points_a: np.ndarray
    :param points_b: 2nd point cloud
    :type points_b: np.ndarray
    :return: normalised points clouds, scale factor & translations
    """

    translate_a = np.mean(points_a, axis=0)
    translate_b = np.mean(points_b, axis=0)
    a_demean = points_a - translate_a
    b_demean = points_b - translate_b

    norm_factor = np.max([np.max(np.abs(a_demean)),
                          np.max(np.abs(b_demean))])

    a_normalised = a_demean / norm_factor
    b_normalised = b_demean / norm_factor

    scale_matrix = create_scaling_matrix(norm_factor)
    translate_a_matrix = create_translation_matrix(translate_a)
    translate_b_matrix = create_translation_matrix(translate_b)

    return a_normalised, b_normalised, scale_matrix, \
         translate_a_matrix, translate_b_matrix

def set_rotnode(limits_degrees) -> ROTNODE:
    """ Setup a ROTNODE with upper/lower rotation limits"""

    lower_degrees = limits_degrees[0]
    upper_degrees = limits_degrees[1]

    l_rads = lower_degrees * 3.14 / 180
    u_rads = upper_degrees * 3.14 / 180

    r_node = ROTNODE()

    r_node.a = l_rads
    r_node.b = l_rads
    r_node.c = l_rads
    r_node.w = u_rads - l_rads

    return r_node

def set_transnode(trans_limits) -> TRANSNODE:
    """ Setup a TRANSNODE with upper/lower limits"""

    t_node = TRANSNODE()

    t_node.x = trans_limits[0]
    t_node.y = trans_limits[0]
    t_node.z = trans_limits[0]

    t_node.w = trans_limits[1] - trans_limits[0]

    return t_node

class RigidRegistration(rr.RigidRegistration):
    """
    Class that uses GoICP implementation to register fixed/moving clouds.
    At the moment, we are just relying on all default parameters.
    :param dt_size: Nodes per dimension of distance transform
    :param dt_factor: GoICP distance transform factor
    TODO: rest of params
    """
    #pylint:disable=dangerous-default-value
    def __init__(self,
                 dt_size: int = 200,
                 dt_factor: float = 2.0,
                 normalise: bool = True,
                 num_moving_points: int = 1000,
                 rotation_limits=[-45, 45],
                 trans_limits=[-0.5, 0.5]):

        r_node = set_rotnode(rotation_limits)
        t_node = set_transnode(trans_limits)

        self.goicp = GoICP()
        self.goicp.setDTSizeAndFactor(dt_size, dt_factor)
        self.goicp.setInitNodeRot(r_node)
        self.goicp.setInitNodeTrans(t_node)

        self.normalise = normalise
        self.num_moving_points = num_moving_points

    def register(self,
                 moving_cloud: np.ndarray,
                 fixed_cloud: np.ndarray) -> np.ndarray:
        """
        Uses GoICP library, wrapped in scikit-surgerygoicp.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Mx3] moving point cloud.
        :param normalise: If true, data will be centred around 0 and normalised.
        :param num_moving_points: How many points to sample from moving cloud \
            if 0, use all points
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """

        LOGGER.info("Fixed cloud shape %s", fixed_cloud.shape)
        LOGGER.info("Moving cloud shape %s", moving_cloud.shape)

        if self.normalise:
            fixed_cloud, moving_cloud, scale, trans_fixed, trans_moving = \
                demean_and_normalise(fixed_cloud, moving_cloud)

        if self.num_moving_points > 0:
            n_moving = moving_cloud.shape[0]

            if self.num_moving_points > n_moving:
                self.num_moving_points = n_moving

            idxs = np.random.choice(range(n_moving),
                                    self.num_moving_points,
                                    replace=False)
            moving_cloud = moving_cloud[idxs, :]

        Nm, a_points = numpy_to_POINT3D_array(fixed_cloud)
        Nd, b_points = numpy_to_POINT3D_array(moving_cloud)

        self.goicp.loadModelAndData(Nm, a_points, Nd, b_points)
        self.goicp.BuildDT()

        residual = self.goicp.Register()

        opt_rot = self.goicp.optimalRotation()
        opt_trans = self.goicp.optimalTranslation()

        fixed_to_moving = np.identity(4)

        for i in range(3):
            for j in range(3):
                fixed_to_moving[i][j] = opt_rot[i][j]

        fixed_to_moving[0][3] = opt_trans[0]
        fixed_to_moving[1][3] = opt_trans[1]
        fixed_to_moving[2][3] = opt_trans[2]

        # Get the transform back in un-normalised space
        # T = Tf * S * GoICP * inv(S) * inv(Tm)
        # Tf/Tm are translation matrices, when demeaning fixed/moving points
        # S is scaling matrix when normalising fixed/moving
        #  GoICP is resultant matrix from GoICP.register()
        if self.normalise:

            moving_to_fixed = trans_fixed @ \
                  scale @ \
                    fixed_to_moving @ \
                         np.linalg.inv(scale) @ \
                             np.linalg.inv(trans_moving)

        return residual, moving_to_fixed
