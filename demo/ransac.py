import h5py
import numpy as np
import argparse
import os
import glob
import pyransac
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
from util import load_h5, save_h5
from learnedmatcher import LearnedMatcher

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='match for ransac_workshop')
parser.add_argument('--dataset_path', type=str, default='/data1/zjh/RANSAC-Tutorial-Data-EF/RANSAC-Tutorial-Data/val/',
  help='datapath_corr.')
parser.add_argument('--dump_path',type=str,default='old_fundamental',help='dump matches and estimeted F/E')

'''network config'''
parser.add_argument('--model_path', type=str, default='/home/jhzhang/IMW2020/OANet-Net2/core/log/main.py/train/model_best.pth',
  help='pretrained model path')
parser.add_argument('--inlier_th', type=float, default=1,
  help='inlier threshold for network output')
parser.add_argument('--net_depth', type=int, nargs='+', default=[1,1,1,1,1,1], help=""
    "number of layers. Default: [3, 3]")
parser.add_argument("--clusters", type=int, default=192, help=""
    "cluster number in OANet. Default: 500")
parser.add_argument("--bottleneck", type=int, default=24, help=""
    "dim in bottleneck layer. If -1, not use bottleneck layer")
parser.add_argument("--cat", type=str2bool, default=True, help=""
    "concat or add")
parser.add_argument('--use_ratio', type=int, default=2,
  help='use ratio test in network')
parser.add_argument('--use_mutual', type=int, default=0,
  help='use mutual check in network')
parser.add_argument('--use_bipartite', type=str2bool, default=False,
  help='use bipartite in network')
parser.add_argument('--use_fundamental',type=str2bool,default=True,
  help='estimate fundamental')

'''ransac config'''
parser.add_argument('--ransac_th', type=float, default=0.75,
  help='inlier threshold (px) in RANSAC variants')
parser.add_argument('--ransac_iter', type=int, default=100000,
  help='inlier threshold (px) in RANSAC variants')

parser.add_argument('--use_cpu', type=str2bool, default=False,
  help='use cpu if you are poor')
parser.add_argument('--num_cores', type=int, default=4,
  help='nums for parallel')

def compute_matches(matcher,post_estimator,corr,sides,K1=None,K2=None):
    matches,E_hat,logits = matcher.infer(corr,sides,K1,K2)
    matches=matches.cpu().numpy()
    E_post, mask = post_estimator(matches[:,:2], matches[:,2:4])
    matches_post = matches[mask,:]

    E_hat=E_hat/np.linalg.norm(E_hat)
    E_post=E_post/np.linalg.norm(E_post)

    return matches_post.astype(np.double),E_post.astype(np.double),matches.astype(np.double),E_hat.astype(np.double),logits.astype(np.double)


if __name__ == "__main__":
    args = parser.parse_args()
    seqs = os.listdir(args.dataset_path)
    matcher = LearnedMatcher(args.model_path, args.net_depth, args.clusters, args.bottleneck, args.cat, args.inlier_th,args.use_ratio,args.use_mutual, use_cpu=args.use_cpu, use_bipartite=args.use_bipartite,fundamental=args.use_fundamental)

    post_estimator = partial(pyransac.findFundamentalMatrix, px_th=args.ransac_th, max_iters = args.ransac_iter)
    if not os.path.exists(args.dump_path):
        os.mkdir(args.dump_path)
    for seq in seqs:
        if not os.path.exists(os.path.join(args.dump_path,seq)):
            os.mkdir(os.path.join(args.dump_path,seq))
        print('---'+str(seq)+'---')
        corr_path = os.path.join(args.dataset_path, seq,'matches.h5')
        side_path=os.path.join(args.dataset_path,seq,'match_conf.h5')
        intrinsic_path = os.path.join(args.dataset_path,seq,'K1_K2.h5')

        corrs,sides,intrinsics = load_h5(corr_path),load_h5(side_path),None if args.use_fundamental else load_h5(intrinsic_path)
        key_list=list(corrs.keys()) 
        matches_dict = {}
        match_fun = partial(compute_matches, matcher=matcher, post_estimator=post_estimator)   
        
        results = Parallel(n_jobs=args.num_cores)(delayed(match_fun)(
            corr=np.asarray(corrs[key]),sides=np.asarray(sides[key]) ,K1=None if args.use_fundamental else np.asarray(intrinsics[key])[0,0],K2=None if args.use_fundamental else np.asarray(intrinsics[key])[0,1]) for key in tqdm(key_list))
        
        #form dict_to_save
        dictsave_corr_post,dictsave_e_post,dictsave_corr,dictsave_e_weighted,dictsave_score={},{},{},{},{}
        for i in range(len(key_list)):
            key=key_list[i]
            corr_post,e_post,corr_th,e_weighted,score=results[i][0],results[i][1],results[i][2],results[i][3],results[i][4]
            dictsave_corr_post[key],dictsave_e_post[key],dictsave_corr[key],dictsave_e_weighted[key],dictsave_score[key]=corr_post,e_post,corr_th,e_weighted,score
        corr_post_path,e_post_path,corr_th_path,e_weighted_path,score_path=os.path.join(args.dump_path,seq,'corr_post.h5'),os.path.join(args.dump_path,seq,'E_post.h5' if not args.use_fundamental else 'F_post.h5'),os.path.join(args.dump_path,seq,'corr_th.h5'),os.path.join(args.dump_path,seq,'E_weighted.h5' if not args.use_fundamental else 'F_weighted.h5'),os.path.join(args.dump_path,seq,'score.h5')
        save_h5(dictsave_corr_post,corr_post_path),save_h5(dictsave_e_post,e_post_path),save_h5(dictsave_corr,corr_th_path),save_h5(dictsave_e_weighted,e_weighted_path),save_h5(dictsave_score,score_path)
