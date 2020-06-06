# Requirements
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). 

For DEGENSAC(https://github.com/vcg-uvic/image-matching-benchmark):
```
pip install git+https://github.com/ducha-aiki/pyransac
```
Other dependencies should be easily installed through pip or conda.

# Example
For a quick start, set the dataset_path in demo/run_bash.sh as the path for test data, run bash run_ransac.sh. A dump folder named 'fundamental' will be created with corresponding inlier corr files and estimated F. For the estimation of E, please change the dump_path, model_path in run_ransac.sh and set use_fundamental=false, this requires a 'K1_K2.h5' file for every sequence. 

There are five files for each test sequence, <br/>

'score.h5':    Prediceted confidence score for each correspondence. <br/>
'corr_th.h5':    Inlier correspondences by applying a thresholding(>1 by default) on the aforementioned confidence score. <br/>
'E/F_weighted':    E/F estimated with weighted 8-point algorithm.<br/>
'corr_post':    Inlier correspondence surviving from score thresholding and degensac.<br/>
'E/F_post':    E/F estimated with degensac using corr_th.<br/>

In most cases, 'E/F_post' are the most precise.

For the parsing and evaluation, please refer to demo/eval_ef.py
