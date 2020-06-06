import torch
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
from collections import namedtuple
import sys
from oan import OANet

        
class LearnedMatcher(object):
    def __init__(self, model_path, net_depth, clusters, bottleneck, cat, inlier_threshold=0, use_ratio=2, use_mutual=2, geo_dist=1e-4, use_cpu=False, use_bipartite=False,fundamental=False):
        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = net_depth
        self.default_config['clusters'] = clusters
        self.default_config['bottleneck'] = bottleneck
        self.default_config['cat'] = cat
        self.default_config['use_ratio'] = use_ratio
        self.default_config['use_mutual'] = use_mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = inlier_threshold
        self.default_config['pos_enc'] = 0
        self.default_config['use_att1'] = False
        self.default_config['use_gn'] = False
        self.default_config['use_att2'] = False
        self.default_config['lg'] = True
        self.default_config['head'] = 1
        self.default_config['softmax_scale']=False
        self.default_config['use_fundamental']=fundamental
        self.default_config = namedtuple("Config", self.default_config.keys())(*self.default_config.values())

        self.model = OANet(self.default_config)
        self.device = torch.device('cuda') if not use_cpu else torch.device('cpu')
        self.use_bipartite = use_bipartite
        self.fundamental=fundamental

        print('load model from ' +model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def normalize_kpts(self, kpts):
        x_mean =torch.mean(kpts, dim=0)
        dist = kpts - x_mean
        meandist = torch.sqrt((dist**2).sum(axis=1)).mean()
        scale = 1.414 / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * torch.from_numpy(np.asarray([T[0, 0], T[1, 1]])).unsqueeze(0).to(self.device) + torch.from_numpy(np.asarray([T[0, 2], T[1, 2]])).unsqueeze(0).to(self.device)
        return nkpts.float(), torch.from_numpy(T).to(self.device).float()

    def normalize_intrinsic(self,kpts,K):
        homo_x=torch.cat([kpts,torch.ones(len(kpts),1).to(self.device)],dim=-1)
        norm_x=torch.matmul(K.inverse(),homo_x.transpose(0,1)).transpose(0,1)
        return norm_x[:,:2]

    def episym(self, x1, x2, F):
        num_pts = x1.shape[0]
        x1 = torch.cat([x1, x1.new_ones(num_pts,1)], dim=-1).reshape(num_pts,3,1)
        x2 = torch.cat([x2, x2.new_ones(num_pts,1)], dim=-1).reshape(num_pts,3,1)
        F = F.reshape(1,3,3)
        x2Fx1 = torch.matmul(x2.transpose(1,2), torch.matmul(F, x1)).reshape(num_pts)
        Fx1 = torch.matmul(F,x1).reshape(num_pts,3)
        Ftx2 = torch.matmul(F.transpose(1,2),x2).reshape(num_pts,3)
        ys = x2Fx1.abs() * (
                1.0 / (Fx1[:, 0]**2 + Fx1[:, 1]**2 + 1e-15).sqrt() +
                1.0 / (Ftx2[:, 0]**2 + Ftx2[:, 1]**2 + 1e-15).sqrt()) / 2
        return ys


    def infer(self, corr, sides,K1=None,K2=None):
        with torch.no_grad():
            corr=torch.from_numpy(corr).to(self.device).float()
            sides=torch.from_numpy(sides).to(self.device).float()
            if self.fundamental:
                x1, T1 = self.normalize_kpts(corr[:,:2])
                x2, T2 = self.normalize_kpts(corr[:,2:4])
            else:
                K1,K2=torch.from_numpy(K1).to(self.device).float(),torch.from_numpy(K2).to(self.device).float()
                x1,x2=self.normalize_intrinsic(corr[:,:2],K1),self.normalize_intrinsic(corr[:,2:4],K2)
                T1, T2 =torch.eye(3).to(self.device),torch.eye(3).to(self.device)
            norm_corr, sides = torch.cat([x1,x2],dim=-1).unsqueeze(0).unsqueeze(0), sides.unsqueeze(0)
            data = {}
            data['xs'] =norm_corr
            data['sides'] = sides.unsqueeze(-1)
            
            y_hat, e_hat = self.model(data)
            y, e_hat = y_hat[-1][0, :].cpu().numpy(), e_hat[-1]
            if self.use_bipartite:
                data['xs'] = data['xs'][:,:,:, [2,3,0,1]]
                y_hat2, e_hat2 = self.model(data)
                y2 = y_hat2[-1][0, :].cpu().numpy()
                y = np.minimum(y, y2)

            e_hat = torch.matmul(torch.matmul(T2.transpose(0,1), e_hat.reshape(3,3)),T1).reshape(-1,9)
            F = (e_hat / torch.norm(e_hat, dim=1, keepdim=True)).reshape(3,3)
            inlier_idx = np.where(y > self.default_config.inlier_threshold)
         
        if len(inlier_idx)<8:
            inlier_idx=np.argpartition(-y,8)[:8]
        matches=corr[inlier_idx]
        if not self.fundamental:
            matches=norm_corr.reshape(-1,4)[inlier_idx]

        return matches,F.cpu().numpy(),y


