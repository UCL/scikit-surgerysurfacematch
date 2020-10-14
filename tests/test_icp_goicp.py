import pytest
import numpy as np

model_data = np.loadtxt('tests/data/icp/model_rand.txt')
target_data = np.loadtxt('tests/data/icp/data_rand.txt')

import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.algorithms.goicp_registration as goicp

def transform_points(points, transform):
    out_points = \
        np.matmul(
            transform[0:3, 0:3], np.transpose(points)) \
        + transform[0:3, 3].reshape((3, 1))

    out_points = np.transpose(out_points)

    return out_points

def test_icp_reg():
    icp_reg = pir.RigidRegistration()
    residual, transform = icp_reg.register(model_data, target_data)

    assert residual < 0.1


def test_goicp_reg():

    fixed = np.loadtxt('tests/data/icp/rabbit_full.xyz')
    moving = np.loadtxt('tests/data/icp/rabbit_partial.xyz')

    goicp_reg = goicp.RigidRegistration()

    # Data already normalsied
    residual, moving_to_fixed = goicp_reg.register(moving, fixed)
    out_points = transform_points(moving, moving_to_fixed)

    np.savetxt('tests/output/goicp_bunny.xyz', out_points)

    assert residual < 0.5

def test_normalise():

    # Create data between 0 and 2
    points_a = 2 * np.random.rand(100,3)
    points_b = 2 * np.random.rand(100,3)

    norm_a, norm_b, _, _, _ = goicp.demean_and_normalise(points_a, points_b)

    tol = 0.01
    assert np.abs(np.mean(norm_a)) < tol
    assert np.abs(np.mean(norm_b)) < tol

    assert np.max(norm_a) <= 1
    assert np.max(norm_b) <= 1
    assert np.min(norm_a) >= -1
    assert np.min(norm_b) >= -1


def test_goicp_known_transform():
    """ Transform the model data by a known matrix - we should get the inverse
    of the matrix back as the result. """
    
    fixed = np.loadtxt('tests/data/icp/rabbit_full.xyz')
    fixed_to_moving = np.array([[1, 0, 0, 1],
                                        [0, 0, 1, 2],
                                        [0, -1, 0, 3],
                                        [0, 0, 0, 1]])

    moving = transform_points(fixed, fixed_to_moving)
    goicp_reg = goicp.RigidRegistration()
    residual, moving_to_fixed = goicp_reg.register(moving, fixed)


    assert np.allclose(moving_to_fixed, np.linalg.inv(fixed_to_moving), atol=1e-3)