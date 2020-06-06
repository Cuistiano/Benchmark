import h5py
import cv2
import torch
from torch.utils.data import Dataset
import sys
import numpy as np
from utils import get_episym

# torch.multiprocessing.set_start_method("spawn")


class H5DataReader:
    def __init__(self, path, use_fundamental):

        ## If this crashes, try swmr=False
        self.mapping = {
            # "F": h5py.File(f"{path}/Fgt.h5", "r", libver="latest", swmr=True),
            'R': h5py.File(f"{path}/R.h5", "r", libver="latest", swmr=True),
            'T': h5py.File(f"{path}/T.h5", "r", libver="latest", swmr=True),
            'K1_K2': h5py.File(f"{path}/K1_K2.h5", "r", libver="latest", swmr=True),
            "matches": h5py.File(f"{path}/matches.h5", "r", libver="latest", swmr=True),
            "confidence": h5py.File(
                f"{path}/match_conf.h5", "r", libver="latest", swmr=True
            ),
        }
        if not use_fundamental:
            self.mapping['E'] = h5py.File(f"{path}/Egt.h5", "r", libver="latest", swmr=True)
        else:
            self.mapping['F'] = h5py.File(f"{path}/Fgt.h5", "r", libver="latest", swmr=True)

        self.path = path
        self.keys = self.__get_h5_keys(self.mapping["confidence"])
        self.num_keys = len(self.keys)
        self.use_fundamental = use_fundamental

    def __get_h5_keys(self, file):
        return [key for key in file.keys()]

    def __uv2xy(self, uv, K):
        xy = (uv - np.asarray([[K[0,2], K[1,2]]])) / np.asarray([[K[0,0], K[1,1]]])
        return xy

    def __correctMatches(self, e_gt):
        step = 0.1
        xx,yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
        # Points in first image before projection
        pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
        # Points in second image before projection
        pts2_virt_b = np.float32(pts1_virt_b)
        pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1,-1,2), pts2_virt_b.reshape(1,-1,2)

        pts1_virt_b, pts2_virt_b = cv2.correctMatches(e_gt.reshape(3,3), pts1_virt_b, pts2_virt_b)

        return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()

    def __hartlynorm(self, x):
        x_mean = np.mean(x, axis=0)
        dist = x - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
        return x, T

    def __load_h5_key(self, mapping, key):
        img_1, img_2 = key.split('-')
        R_i, R_j = mapping['R'][img_1][:], mapping['R'][img_2][:]
        T_i, T_j = mapping['T'][img_1][:], mapping['T'][img_2][:]
        dR = np.dot(R_j, R_i.T)
        t_i, t_j = T_i.reshape([3, 1]), T_j.reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)

        K1, K2 = mapping['K1_K2'][key][:][0]
        uv = mapping['matches'][key][:]
        x1, x2 = uv[:,:2], uv[:,2:4]
        x1, x2 = self.__uv2xy(x1, K1), self.__uv2xy(x2, K2)

        geod_d = get_episym(x1, x2, dR, dt)
        ys = geod_d.reshape(-1,1)

        side = mapping['confidence'][key][:].reshape(-1,1)

        if not self.use_fundamental:
            e_gt = (mapping['E'][key][:]).reshape(3,3)
            e_gt = e_gt / np.linalg.norm(e_gt)
        else:
            x1, T1 = self.__hartlynorm(uv[:,:2])
            x2, T2 = self.__hartlynorm(uv[:,2:4])
            F = (mapping['F'][key][:]).reshape(3,3)
            F = np.matmul(np.matmul(np.linalg.inv(T2).T, F), np.linalg.inv(T1))
            e_gt = F / np.linalg.norm(F)

        xs = np.concatenate([x1, x2], axis=1).reshape(1,-1,4)
        pts1_virt, pts2_virt = self.__correctMatches(e_gt.astype('float64'))
        pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float32')

        
        #import pdb;pdb.set_trace()
        if not self.use_fundamental:
            return {'xs':xs, 'ys':ys, 'R':dR, 't':dt, 'side':side, 'virtPt':pts_virt}
        else:
            return {'xs':xs, 'ys':ys, 'R':dR, 't':dt, 'side':side, 'virtPt':pts_virt, 'K1':K1, 'K2':K2, 'T1':T1, 'T2':T2}



    def __getitem__(self, idx):
        return self.__load_h5_key(self.mapping, self.keys[idx]), self.keys[idx]

    def __len__(self):
        return len(self.keys)


class DummyH5Dataset(Dataset):
    def __init__(self, path, use_fundamental):
        self.path = path
        self.reader = None
        self.use_fundamental = use_fundamental

    def __getitem__(self, idx):
        ## This is important to make hdf5 work with multiprocessing
        ## Opening the file in the constructor will lead to crashes
        if self.reader is None:
            self.reader = H5DataReader(self.path, self.use_fundamental)

        data, name = self.reader.__getitem__(idx)
        return data

        ## Do something

    def __len__(self):
        return len(H5DataReader(self.path, self.use_fundamental))


def collate_fn_E(batch):
    #import pdb;pdb.set_trace()
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['Rs'], data['ts'], data['xs'], data['ys'], data['virtPts'], data['sides']  = [], [], [], [], [], []
    for sample in batch:
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        data['virtPts'].append(sample['virtPt'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp, replace=False)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'])

    for key in ['Rs', 'ts', 'xs', 'ys','virtPts', 'sides']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    return data

def collate_fn_F(batch):
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['K1s'], data['K2s'], data['Rs'], \
        data['ts'], data['xs'], data['ys'], data['T1s'], data['T2s'], data['virtPts'], data['sides']  = [], [], [], [], [], [], [], [], [], []
    for sample in batch:
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        data['virtPts'].append(sample['virtPt'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp, replace=False)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            data['sides'].append(sample['side'])

    for key in ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s','virtPts', 'sides']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    return data    
