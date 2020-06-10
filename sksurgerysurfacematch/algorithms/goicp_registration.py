# -*- coding: utf-8 -*-

""" PCL ICP implementation of RigidRegistration interface. """

import numpy as np
from sksurgerygoicppython import  GoICP, POINT3D, ROTNODE, TRANSNODE
import sksurgerysurfacematch.interfaces.rigid_registration as rr

def numpy_to_POINT3D_array(numpy_pointcloud):
    plist = numpy_pointcloud.tolist();
    p3dlist = [];
    for x,y,z in plist:
        pt = POINT3D(x,y,z);
        p3dlist.append(pt);
    return numpy_pointcloud.shape[0], p3dlist;

class GoICPRegistration(rr.RigidRegistration):
    """
    Class that uses GoICP implementation to register fixed/moving clouds.
    At the moment, we are just relying on all default parameters.
    """
    def register(self,
                 fixed_cloud: np.ndarray,
                 moving_cloud: np.ndarray
                 ):
        """
        Uses GoICP library, wrapped in scikit-surgerygoicp.

        :param fixed_cloud: [Nx3] fixed point cloud.
        :param moving_cloud: [Mx3] moving point cloud.
        :return: [4x4] transformation matrix, moving-to-fixed space.
        """

        goicp = GoICP();
        Nm, a_points = numpy_to_POINT3D_array(moving_cloud)
        Nd, b_points = numpy_to_POINT3D_array(fixed_cloud)
        goicp.loadModelAndData(Nm, a_points, Nd, b_points)
        goicp.setDTSizeAndFactor(300, 2.0)
        goicp.BuildDT()
        residual = goicp.Register()

        opt_rot = goicp.optimalRotation()
        opt_trans = goicp.optimalTranslation()

        print(opt_rot)
        print(opt_trans)
        print('-----')
        transform = 1
        return residual, transform

