import os, json, random, torch, sys
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

sys.path.append("../")

from utils import image
from utils.data import NumpyImageLoader
from utils.models import MobileNetV2



pools = Pool(processes=cpu_count())

# folder='/home/atsg/PycharmProjects/image_forgery/Image-Forgery-using-Deep-Learning/cmnd'
#
# IMAGE_FILE='/home/atsg/only_id.jpg'
#
# if(IMAGE_FILE==''):
#     files = glob(folder+"/*.*")
#     n_files = len(files)
#     random_idx = random.choice(list(range(n_files)))
#     IMAGE_FILE = files[random_idx]
#
# print (IMAGE_FILE)
# GT = IMAGE_FILE.split("/")[-1][:2]
# GT = "Positive" if GT=="Tp" else "Negative"

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def detect_image(filename):
    # Load data
    img = np.array(Image.open(filename).convert(params["channel"]))
    coords, _, _ = image.slide2d(sz=img.shape[:2], K=64, S=32)
    patches = image.crop_patches(img=img, coords=coords, patch_sz=64)
    loader = NumpyImageLoader(ndarray_data=patches, batch_size=16, n_workers=cpu_count(), pin_memory=True,
                              shuffle=False).loader

    # Predict
    softmaxs = []
    model.eval()
    for X in loader:
        X = X[0].to(DEVICE)
        logits = model(X)
        softmaxs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    softmaxs = np.concatenate(softmaxs, axis=0)

    # Post-processing
    labels = image.post_process(softmaxs[:, 1], coords, 8, params["threshold"], 32, pools=pools)
    decision = image.fusion(labels)
    return decision  #1 fake, 0 real


params={}
params["channel"] = "YCbCr"
params["threshold"] = 0.645
params["training_log_dir"] = "../backup/MBN2-mod-YCbCr/checkpoints/"

MODEL_FILE = os.path.join(params["training_log_dir"], "model.ckpt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV2(n_classes=2).to(device=DEVICE)
model.load(model_file=MODEL_FILE)

data_dir='/media/atsg/Data/datasets/image_forgery/KYC'
all_dir=get_list_dir_in_folder(data_dir)
real_dir=[]
fake_dir=['Fake']

for dir in all_dir:
    if dir !='Fake':
        real_dir.append(dir)



#
# for dir in real_dir:
#     print (dir),
#     count=0
#     true_pred=0
#     all_files= get_list_file_in_folder(os.path.join(data_dir,dir))
#     for file in all_files:
#         count+=1
#         pred=detect_image(os.path.join(data_dir,dir,file))
#         if(pred==0):
#             true_pred+=1
#     print ('True pred:',true_pred,', Total:',count,', Accuracy of real image: ',float(true_pred)/float(count))



for dir in fake_dir:
    print (dir),
    count=0
    true_pred=0
    all_files= get_list_file_in_folder(os.path.join(data_dir,dir))
    for file in all_files:
        count+=1
        pred=detect_image(os.path.join(data_dir,dir,file))
        if(pred==1):
            true_pred+=1
    print ('True pred:',true_pred,', Total:',count,', Accuracy of fake image: ',float(true_pred)/float(count))





#
#
# # Load data
# img = np.array(Image.open(IMAGE_FILE).convert(params["channel"]))
# coords, _, _ = image.slide2d(sz=img.shape[:2], K=64, S=32)
# patches = image.crop_patches( img=img, coords=coords, patch_sz=64)
# loader = NumpyImageLoader(ndarray_data=patches, batch_size=16, n_workers=cpu_count(), pin_memory=True, shuffle=False).loader
#
# # Predict
# softmaxs = []
# model.eval()
# for X in loader:
#     X = X[0].to(DEVICE)
#     logits = model(X)
#     softmaxs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
# softmaxs = np.concatenate(softmaxs, axis=0)
#
# # Post-processing
# labels = image.post_process(softmaxs[:,1], coords, 8, params["threshold"], 32, pools=pools)
# decision = image.fusion(labels)
# result=''
# if decision==1:
#     if GT=="Positive":
#         result ="Prediction: Positive ==> True"
#     else:
#         result ="Prediction: Positive ==> False"
#     result='Fake'
# else:
#     if GT=="Positive":
#         result ="Prediction: Negative ==> False"
#     else:
#         result ="Prediction: Negative ==> True"
#     result='Real'
#
# # Draw score_map
# img_PIL = Image.open(IMAGE_FILE).convert("RGB")
# score_map = image.reconstruct_heatmap(softmaxs[:,1], coords, img.shape, 64)
# plt.figure(figsize=(16,8))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img_PIL)
# plt.title(IMAGE_FILE)
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(score_map, cmap="gray")
# plt.title(result)
# plt.axis('off')
# plt.show()