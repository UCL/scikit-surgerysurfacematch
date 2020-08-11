import pytest
import numpy as np

model_data = np.loadtxt('tests/data/icp/model_rand.txt')
target_data = np.loadtxt('tests/data/icp/data_rand.txt')

import sksurgerysurfacematch.algorithms.pcl_icp_registration as pir
import sksurgerysurfacematch.algorithms.goicp_registration as goicp

def test_icp_reg():
    icp_reg = pir.RigidRegistration()
    residual, transform = icp_reg.register(model_data, target_data)

    assert residual < 0.1


def test_goicp_reg():
    goicp_reg = goicp.RigidRegistration()
    residual, transform = goicp_reg.register(model_data, target_data)

    print(residual, transform)
    assert residual < 0.1

def test_normalise():

    # Create data between 0 and 2
    points_a = 2 * np.random.rand(100,3)
    points_b = 2 * np.random.rand(100,3)

    norm_a, norm_b = goicp.demean_and_normalise(points_a, points_b)

    tol = 0.01
    assert np.abs(np.mean(norm_a)) < tol
    assert np.abs(np.mean(norm_b)) < tol

    assert np.max(norm_a) <= 1
    assert np.max(norm_b) <= 1
    assert np.min(norm_a) >= -1
    assert np.min(norm_b) >= -1
