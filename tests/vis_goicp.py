import numpy as np
import pptk
import sksurgerysurfacematch.algorithms.goicp_registration as goicp

model_data = np.loadtxt('tests/data/icp/model_bunny.txt')
target_data = np.loadtxt('tests/data/icp/data_bunny.txt')

all_xyz = np.vstack((model_data, target_data))

white = np.ones(model_data.shape)
black = np.zeros(target_data.shape)
all_rgb = np.vstack((black, white))

x = pptk.viewer(all_xyz, all_rgb)
x.wait()

print('Starting GoICP')
goicp_reg = goicp.RigidRegistration(dt_size=100, dt_factor=2.0)
goicp_reg.goicp.MSEThresh = 0.01
residual, transform = goicp_reg.register(model_data, target_data, False)

print(residual, transform)

moved_cloud = \
    np.matmul(
        transform[0:3, 0:3], np.transpose(target_data)) \
    + transform[0:3, 3].reshape((3, 1))

moved_cloud = np.transpose(moved_cloud)

all_xyz = np.vstack((model_data, moved_cloud))

x = pptk.viewer(all_xyz, all_rgb)
x.wait()
x.clear()