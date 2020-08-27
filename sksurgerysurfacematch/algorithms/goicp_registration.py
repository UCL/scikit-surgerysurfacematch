# -*- coding: utf-8 -*-

""" Go ICP implementation of RigidRegistration interface. """

# pylint:disable=invalid-name, no-name-in-module

import numpy as np
from sksurgerygoicppython import  GoICP, POINT3D
import sksurgerysurfacematch.interfaces.rigid_registration as rr


def numpy_to_POINT3D_array(numpy_pointcloud):
    """ Covert numpy array to POINT3D array suitable for GoICP algorithm."""
    plist = numpy_pointcloud.tolist()
    p3dlist = []
    for x, y, z in plist:
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return numpy_pointcloud.shape[0], p3dlist


def demean_and_normalise(points_a: np.ndarray,
                         points_b: np.ndarray):
    """
    Independently centre each point cloud around 0,0,0, then normalise
    both to [-1,1].

    :param points_a: [description]
    :type points_a: np.ndarray
    :param points_b: [description]
    :type points_b: np.ndarray
    :return: [description]
    :rtype: [type]
    """
    a_demean = points_a - np.mean(points_a, axis=0)
    b_demean = points_b - np.mean(points_b, axis=0)

    norm_factor = np.max([np.max(np.abs(a_demean)),
                          np.max(np.abs(b_demean))])

    a_normalised = a_demean / norm_factor
    b_normalised = b_demean / norm_factor

    return a_normalised, b_normalised


class RigidRegistration(rr.RigidRegistration):
    """
    Class that uses GoICP implementation to register fixed/moving clouds.
    At the moment, we are just relying on all default parameters.
    :param dt_size: Nodes per dimension of distance transform
    :param dt_factor: GoICP distance transform factor
    """

    def __init__(self, dt_size=300, dt_factor=2.0):

        self.goicp = GoICP()
        self.goicp.setDTSizeAndFactor(dt_size, dt_factor)

    def register(self,
                 fixed_cloud: np.ndarray,
                 moving_cloud: np.ndarray,
                 normalise=True
                 ):
        """
        Uses GoICP library, wrapped in scikit-surgerygoicp.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Mx3] moving point cloud.
        :param normalise: If true, data will be centred around 0 and normalised.
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """

        if normalise:
            fixed_cloud, moving_cloud = \
                demean_and_normalise(fixed_cloud, moving_cloud)

        Nm, a_points = numpy_to_POINT3D_array(moving_cloud)
        Nd, b_points = numpy_to_POINT3D_array(fixed_cloud)
        self.goicp.loadModelAndData(Nm, a_points, Nd, b_points)
        self.goicp.BuildDT()
        residual = self.goicp.Register()

        opt_rot = self.goicp.optimalRotation()
        opt_trans = self.goicp.optimalTranslation()

        rigid_transformation = np.identity(4)

        for i in range(3):
            for j in range(3):
                rigid_transformation[i][j] = opt_rot[i][j]

        rigid_transformation[0][3] = opt_trans[0]
        rigid_transformation[1][3] = opt_trans[1]
        rigid_transformation[2][3] = opt_trans[2]

        return residual, rigid_transformation
