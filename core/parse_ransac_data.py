import h5py 
import numpy as np
from utils import np_skew_symmetric



path = '/data1/zjh/RANSAC-Tutorial-Data-EF/RANSAC-Tutorial-Data/train/brandenburg_gate/'
E_file = h5py.File(path+'Egt.h5', 'r', libver="latest")
R_file = h5py.File(path+'R.h5', 'r', libver="latest")
T_file = h5py.File(path+'T.h5', 'r', libver="latest")

keys = [k for k in E_file.keys()]
e0 = E_file[keys[0]][:]
R_i = R_file[keys[0].split('-')[0]][:]
R_j = R_file[keys[0].split('-')[1]][:]
T_i = T_file[keys[0].split('-')[0]][:]
T_j = T_file[keys[0].split('-')[1]][:]

dR = np.dot(R_j, R_i.T)
t_i, t_j = T_i.reshape([3, 1]), T_j.reshape([3, 1])
dt = t_j - np.dot(dR, t_i)

e_gt_unnorm = np.reshape(np.matmul(
    np.reshape(np_skew_symmetric(dt.reshape(1,3)), (3, 3)), np.reshape(dR.astype('float64'), (3, 3))), (3, 3))
e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)

E_saved = e0 / np.linalg.norm(e0)

print(e_gt)
print(E_saved)
