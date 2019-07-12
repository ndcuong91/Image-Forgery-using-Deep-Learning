#!/usr/bin/env python
# coding: utf-8

# # Test the original MobileNetV2 with YCbCr color channel

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import os, json, argparse, torch, sys
import numpy as np
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
sys.path.append("../")

from utils import image
from utils.MobileNetV2_pretrained_imagenet import MobileNetV2
from utils.data import NumpyImageLoader
from utils.metrics import BinaryClassificationMetrics


# # Initial procedure

# In[2]:


# Print parameters

params = {}
params["channel"] = "YCbCr"
params["threshold"] = 0.65
params["test_subset"] = 5

params["patch_test_au_dir"] = "../backup/MBN2-YCbCr/test/au"
params["patch_test_tp_dir"] = "../backup/MBN2-YCbCr/test/tp"

params["training_log_dir"] = "../backup/MBN2-YCbCr/checkpoints/"
MODEL_FILE = os.path.join(params["training_log_dir"], "model.ckpt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params["au_subsets_file"] = "../dataset/au_subsets.json"
params["tp_subsets_file"] = "../dataset/tp_subsets.json"

params["casia2_au"] = "/media/atsg/Data/datasets/casia-dataset/CASIA2/Au"
params["casia2_tp"] = "/media/atsg/Data/datasets/casia-dataset/CASIA2/Tp"


# In[3]:


def check_directories(list_dirs):
    for dir in list_dirs:
        if not os.path.exists(dir):
            print("makedirs", dir)
            os.makedirs(dir)


# In[4]:


# Check directories

list_dirs = [
    params["patch_test_au_dir"],
    params["patch_test_tp_dir"],
]
check_directories(list_dirs)


# # Test on predicted features

# In[5]:


# Create parallel pools

pools = Pool(processes=cpu_count())


# In[6]:


# Get information about files on disk

au_files = glob(os.path.join(params["patch_test_au_dir"], "*.*"))
tp_files = glob(os.path.join(params["patch_test_tp_dir"], "*.*"))
n_au_files, n_tp_files = len(au_files), len(tp_files)
scores_au, scores_tp = [], []


# In[7]:


# Test on authentic images

for i, file in tqdm(enumerate(au_files), total=n_au_files):
    # Load softmaxs and coords from disk
    data = np.load(file).item()
    softmaxs, coords = data["softmaxs"], data["coords"]
    softmaxs = softmaxs[:, 1]

    # Postprocess
    labels = image.post_process(softmaxs, coords, 8, params["threshold"], 32, pools=pools)
    mark = image.fusion(labels)
    scores_au.append(mark)


# In[8]:


# Test on tampered images

for i, file in tqdm(enumerate(tp_files), total=n_tp_files):
    # Load softmaxs and coords from disk
    data = np.load(file).item()
    softmaxs, coords = data["softmaxs"], data["coords"]
    softmaxs = softmaxs[:, 1]

    # Postprocess
    labels = image.post_process(softmaxs, coords, 8, params["threshold"], 32, pools=pools)
    mark = image.fusion(labels)
    scores_tp.append(mark)


# In[9]:


# Print testing metrics

metrics = BinaryClassificationMetrics()
metrics.compute_all(scores_tp, scores_au)
metrics.print_metrics()
# metrics.write_to_file(params["test_result_file"])


# In[10]:


# Close parallel pools

pools.close()
pools.terminate()


# In[ ]:




