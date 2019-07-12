import os, shutil
import cv2

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


data_dir='/media/atsg/Data/datasets/image_forgery/KYC'
all_dir=get_list_dir_in_folder(data_dir)
real_dir=[]
fake_dir=['Fake']
for dir in all_dir:
    if dir !='Fake':
        real_dir.append(dir)

dest_dir='/media/atsg/Data/datasets/image_forgery/kyc_new/tp'

count = 0
for dir in fake_dir:
    print (dir),
    all_files= get_list_file_in_folder(os.path.join(data_dir,dir))
    for file in all_files:
        count+=1
        img=cv2.imread(os.path.join(data_dir,dir,file))
        name=str(count).zfill(4)+'.jpg'
        cv2.imwrite(os.path.join(dest_dir,name),img)
        kk=1
