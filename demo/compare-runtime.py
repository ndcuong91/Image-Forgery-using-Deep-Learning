#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:

import os, torch, sys
from PIL import Image
import numpy as np
from time import time
from torchsummary import summary
from multiprocessing import cpu_count
sys.path.append("../")

from utils import image
from utils.data import NumpyImageLoader


# In[2]:


IMAGE_FILE = "/media/atsg/Data/datasets/casia-dataset/CASIA2/Tp/Tp_D_CND_S_N_ani00073_ani00068_00193.tif"


# # mobilenetv2_orig

# In[3]:


from utils.MobileNetV2_pretrained_imagenet import MobileNetV2

params = {}
params["channel"] = "YCbCr"
params["threshold"] = 0.65
params["training_log_dir"] = "../backup/MBN2-YCbCr/checkpoints/"

MODEL_FILE = os.path.join(params["training_log_dir"], "model.ckpt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV2(n_class=2, input_size=64, width_mult=1.0).to(device=DEVICE)
model.load(model_file=MODEL_FILE)
model.eval()
summary(model, input_size=(3, 64, 64))


# In[4]:


# Load data
img = np.array(Image.open(IMAGE_FILE).convert("YCbCr"))
coords, _, _ = image.slide2d(sz=img.shape[:2], K=64, S=32)
patches = image.crop_patches( img=img, coords=coords, patch_sz=64)
loader = NumpyImageLoader( ndarray_data=patches, batch_size=1, n_workers=cpu_count(), pin_memory=True, shuffle=False).loader

# Measure time
times = []
for X in loader:
    X = X[0].to(DEVICE)
    start_time = time()
    logits = model(X)
    end_time = time()
    times.append(end_time-start_time)
    
print("Runtime: %.1f[ms]" % (1000*sum(times)/len(times)))


# # mobilenetv2_pretrained_imagenet

# In[5]:


from utils.MobileNetV2_pretrained_imagenet import MobileNetV2

params["channel"] = "YCbCr"
params["threshold"] = 0.80
params["training_log_dir"] = "../backup/MBN2-pre-YCbCr/checkpoints/"

MODEL_FILE = os.path.join(params["training_log_dir"], "model.ckpt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV2(n_class=2, input_size=64, width_mult=1.0).to(DEVICE)
model.load(model_file=MODEL_FILE)
model.eval()
summary(model, input_size=(3, 64, 64))


# In[6]:


# Load data
img = np.array(Image.open(IMAGE_FILE).convert("YCbCr"))
coords, _, _ = image.slide2d(sz=img.shape[:2], K=64, S=32)
patches = image.crop_patches( img=img, coords=coords, patch_sz=64)
loader = NumpyImageLoader( ndarray_data=patches, batch_size=1, n_workers=cpu_count(), pin_memory=True, shuffle=False).loader

# Measure time
times = []
for X in loader:
    X = X[0].to(DEVICE)
    start_time = time()
    logits = model(X)
    end_time = time()
    times.append(end_time-start_time)
    
print("Runtime: %.1f[ms]" % (1000*sum(times)/len(times)))


# # MobileNetV2_mod

# In[7]:


from utils.models import MobileNetV2

params["channel"] = "YCbCr"
params["threshold"] = 0.645
params["training_log_dir"] = "../backup/MBN2-mod-YCbCr/checkpoints/"

MODEL_FILE = os.path.join(params["training_log_dir"], "model.ckpt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV2(n_classes=2).to(device=DEVICE)
model.load(model_file=MODEL_FILE)
model.eval()
summary(model, input_size=(3, 64, 64))


# In[8]:


# Load data
img = np.array(Image.open(IMAGE_FILE).convert("YCbCr"))
coords, _, _ = image.slide2d(sz=img.shape[:2], K=64, S=32)
patches = image.crop_patches( img=img, coords=coords, patch_sz=64)
loader = NumpyImageLoader(ndarray_data=patches, batch_size=1, n_workers=cpu_count(), pin_memory=True, shuffle=False).loader

# Measure time
times = []
for X in loader:
    X = X[0].to(DEVICE)
    start_time = time()
    logits = model(X)
    end_time = time()
    times.append(end_time-start_time)
    
print("Runtime: %.1f[ms]" % (1000*sum(times)/len(times)))


# In[ ]:




