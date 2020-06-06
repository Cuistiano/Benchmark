import h5py
import numpy as np
import argparse
import os
import glob
import time
#import pymagsac
import pyransac
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
from io_util import load_h5, save_h5,pose_auc
from learnedmatcher import LearnedMatcher
import cv2
from io_util import quaternion_from_matrix
from tqdm import tqdm

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='eval for ransac_workshop')
parser.add_argument('--dataset_path', type=str, default='/data1/zjh/RANSAC-Tutorial-Data-EF/RANSAC-Tutorial-Data/val/',
  help='datapath_corr.')
parser.add_argument('--dump_path',type=str,default='old_essential',help='dump matches and estimeted F/E')
parser.add_argument('--fundamental',type=str2bool,default=False,help='dump matches and estimeted F/E')


def evaluate_R_t(R_gt, t_gt, R, t):
    t = t.flatten()
    t_gt = t_gt.flatten()
    
    eps = 1e-15
    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))
    
    return err_q, err_t

def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2
    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t



if __name__=="__main__":
    args = parser.parse_args()
    seqs = os.listdir(args.dataset_path)
    err_q_list,err_t_list=[],[]
    for seq in seqs:
        print('---'+str(seq)+'---')
        e_es_path=os.path.join(args.dump_path,seq,'E_post.h5' if not args.fundamental else 'F_post.h5')
        e_gt_path=os.path.join(args.dataset_path,seq,'Egt.h5')
        intrinsic_path = os.path.join(args.dataset_path,seq,'K1_K2.h5')

        R_path,T_path=os.path.join(args.dataset_path,seq,'R.h5'),os.path.join(args.dataset_path,seq,'T.h5')
        corr_es_path=os.path.join(args.dump_path,seq,'corr_post.h5')
        e_gt,e_es,R,T,corr_es,K=load_h5(e_gt_path),load_h5(e_es_path),load_h5(R_path),load_h5(T_path),load_h5(corr_es_path),load_h5(intrinsic_path)
        key_list=list(e_es.keys())
        
        for key in tqdm(key_list):
            img1,img2=key.split('-')[0],key.split('-')[1]
            cur_e_es,R1,R2,T1,T2=np.asarray(e_es[key]),np.asarray(R[img1]),np.asarray(R[img2]),np.asarray(T[img1]),np.asarray(T[img2])
            cur_e_gt=np.asarray(e_gt[key])
            K1,K2=np.asarray(K[key][0,0]),np.asarray(K[key][0,1])
            cur_corr=corr_es[key]
            dR=np.matmul(R2,R1.T)
            dT=T2-np.matmul(dR,T1)
            if args.fundamental:
                cur_e_es=np.matmul(K2.T,np.matmul(cur_e_es,K1))
                cur_corr=(cur_corr-np.asarray([K1[0,2],K1[1,2],K2[0,2],K2[1,2]]))/np.asarray([K1[0,0],K1[1,1],K2[0,0],K2[1,1]])
            err_q,err_t=eval_essential_matrix(cur_corr[:,:2],cur_corr[:,2:4],cur_e_es,dR,dT)
            err_q_list.append(err_q),err_t_list.append(err_t)
            print(err_q*180/np.pi,err_t*180/np.pi)
        
    ths = np.arange(7) * 5
    err_q = np.array(err_q_list) * 180.0 / np.pi
    err_t = np.array(err_t_list) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(err_q, ths)
    t_acc_hist, _ = np.histogram(err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(err_q, err_t), ths)
    num_pair = float(len(err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)

    for i in range(1, len(ths)):
        print('auc_'+str(ths[i])+': '+str(np.mean(qt_acc[:i])))
    err_qt = np.maximum(err_q, err_t)
    exact_auc=pose_auc(err_qt,[5,10,20])
    print('exact_auc: ',exact_auc)
