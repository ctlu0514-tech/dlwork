from cProfile import label
import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as tr
import numpy as np
import random
import SimpleITK as sitk
import csv
import pandas as pd
from torch.utils.data import WeightedRandomSampler,DataLoader
from sklearn.model_selection import KFold,StratifiedKFold
import torch.nn.functional as F
from data_utils_EPE import resize_image
from monai.transforms import EnsureChannelFirst, Compose, RandAffine, RandRotate90, RandFlip, apply_transform, ToTensor, RandGaussianSmooth, RandHistogramShift, GaussianSmooth
from numpy import array

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


def choosezero(image):
    """Preprocess image by handling zero values.
    """
    image[image == 0] = 0.001
    return image
def extract_file_names(serial,folder_path):
    for file_name in os.listdir(folder_path):
        if serial in file_name and "img.nii.gz" in file_name:
            return file_name
def irm_min_max_preprocess(img, low_perc=1, high_perc=99): #异常值处理与归一化操作
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """
    image = choosezero(img)
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

class bal_Dataset(data.Dataset):
    """
    dataloader for ovarian cancer image data
    """
    def __init__(self, image_root, csv_file, transforms=None, task='label',is_train=True):
        self.root = image_root
        self.datas = []
        self.dr = []
        self.transforms = transforms
        label = ['0','1']
        Serial = ['ADC', 'DWI', 'T2FS', 'CT', 'PET']
        #Serial = ['ADC', 'DWI', 'T2FS']
        #Serial = [ 'CT', 'PET']
        for row in csv_file:
            patient_id = row['id']
            patient_label = row['Label']
            #patient_age = row['Age']
            #patient_height = row['height']
            #patient_weight = row['weight']
            #patient_tpsa1 = row['first_tPSA']
            #patient_tpsa2 = row['pre_tPSA']
            images = []
            for serial in Serial:
                image_file1 = os.path.join(image_root, patient_id)
                #print(image_file1)
                niiname = extract_file_names(serial, image_file1)
                #print(niiname)
                image_file = os.path.join(image_file1, niiname)
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_file)))
                # image = np.transpose(image,(1,2,0))
                # output_shape = [8,64,64]
                # ori_size = image.shape
                # image = zoom(image, (output_shape[0]/ori_size[0], output_shape[1]/ori_size[1], output_shape[2]/ori_size[2]))
                images.append(image)
            # patient_invol = row['cortical involvement']
          
            # meta = np.array((row['volume']), dtype=np.float32).reshape(1,)
            # meta = np.array((row['nihss'],row['earlyseizures'],row['volume'],row['majorlocation']), dtype=np.float32)
            # meta = np.array((row['majorlocation']), dtype=np.float32).reshape(1,)
            
            
            # image_path = os.path.join(image_root, 'image', f'{patient_id}.nii')
            # image_crop_path = os.path.join(image_root, 'image_crop', f'{patient_id}.nii')
            
            if task =='label':
                label = float(patient_label)
                # invol = float(patient_invol) # str change to float for having attribute 'cuda'

            # image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
            # image_crop = sitk.GetArrayFromImage(sitk.ReadImage(str(image_crop_path)))
            
            self.dr.append(int(label))

            patient = dict(id=patient_id,  images = images, label = label)
            '''
            patient = dict(id=patient_id,  images = images, age=patient_age, height=patient_height, weight=patient_weight,
                      tpsa1=patient_tpsa1, tpsa2=patient_tpsa2, label = label)
            '''
            self.datas.append(patient)


    def __getitem__(self, index, is_train=True):
        
        _patient = self.datas[index]
        images1 = _patient["images"]

        images = []
        # image_crop = _patient["image_crop"]
        # image_crop = image_crop.astype(np.float32)
        iamges_ = [np.float32(item) for item in images1]
        label = _patient["label"]
        #age = _patient["age"]
        #age = float(age)
        #age = torch.tensor(age)
        #height = _patient["height"]
        #height = float(height)
        #height = torch.tensor(height)
        #weight = _patient["weight"]
        #weight = float(weight)
        #weight = torch.tensor(weight)
        #tpsa1 = _patient["tpsa1"]
        #tpsa1 = float(tpsa1)
        #tpsa1 = torch.tensor(tpsa1)
        #tpsa2 = _patient["tpsa2"]
        #tpsa2 = float(tpsa2)
        #tpsa2 = torch.tensor(tpsa2)
        # meta = _patient["meta_data"]
        if self.transforms is not None and label == 1:
            # image = apply_transform(self.transforms, img_normalize)
            # image = torch.unsqueeze(image, dim=0)
            for item in iamges_:
                image = apply_transform(self.transforms, item)
                # image = torch.unsqueeze(image, dim=0)
                images.append(image)

            # image_crop = apply_transform(self.transforms, image_crop)
        else:
            # image = torch.tensor(img_normalize)
            # image = torch.unsqueeze(image, dim=0)
            for item in iamges_:

                image = torch.tensor(item)
                # image = torch.unsqueeze(image, dim=0)
                images.append(image)
            # image_crop = torch.tensor(image_crop)
            # image_crop = torch.unsqueeze(image_crop, dim=0)
        img = resize_image(images, (8, 64, 64))
        img = np.array(img)
        img = img.astype(np.float32)
        img_normalize = irm_min_max_preprocess(img)
        dr = self.dr[index]
        return  dict(patient_id = _patient["id"],
                     all=img_normalize,
                     label = label
                    )
        '''
        return  dict(patient_id = _patient["id"],
                     ADC=img_normalize[0:1], DWI=img_normalize[1:2], T2FS=img_normalize[2:3],CT=img_normalize[3:4],
                     PET=img_normalize[4:5],
                     all=img_normalize,
                     label = label
                    )
        
        return  dict(patient_id = _patient["id"],age=age,height=height,weight=weight,tpsa1=tpsa1,tpsa2=tpsa2,
                     ADC=img_normalize[0:1], DWI=img_normalize[1:2], T2FS=img_normalize[2:3],CT=img_normalize[3:4],
                     PET=img_normalize[4:5],
                     all=img_normalize,
                     label = label
                    )
        return  dict(patient_id = _patient["id"],
                 ADC=img_normalize[0:1], DWI=img_normalize[1:2], T2FS=img_normalize[2:3],
                 all=img_normalize,
                 label = label
                )

        return  dict(patient_id = _patient["id"],
             CT=img_normalize[0:1], PET=img_normalize[1:2],
             all=img_normalize,
             label = label
            )
    '''
    def __len__(self):
        return len(self.datas)


def get_dataset(data_root, cv_data_source, train_mode, seed=42, fold_number=0):

    image_path = os.path.join(data_root, 'Dataset_five_N4_gamma')
    csv_path = os.path.join(data_root, 'Train_ISUP.csv')

    # fuyi_image_path = os.path.join(data_root, 'Fuyi')
    # fuyi_csv_path = os.path.join(data_root, 'Fuyi/fuyi.csv')
    #
    # tongji_image_path = os.path.join(data_root, 'Tongji')
    # tongji_csv_path = os.path.join(data_root, 'Tongji/tongji.csv')

    csv_file_1, csv_file_2 = [], []
    with open(csv_path, encoding="gbk", mode="r") as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            csv_file_1.append(row)
            #if row['Label'] == '0' :
                #csv_file_1.append(row)
            #elif row['Label'] == '1':
                #csv_file_2.append(row)

    # tongji_csv_file = []
    # with open(tongji_csv_path, encoding="utf-8", mode="r") as f:
    #     tongji_csv_reader = csv.DictReader(f)
    #     for row in tongji_csv_reader:
    #         tongji_csv_file.append(row)
            
    # Cross Validation
    if train_mode == 'Cross_Validation':
        # if cv_data_source == 'fuyi':
        # csv_file = csv_file
        # image_path = image_path
        # elif cv_data_source == 'tongji':
        #     csv_file = tongji_csv_file
        #     image_path = tongji_image_path
        # Cross Validation
        kfold = KFold(10, shuffle=True, random_state=seed)
        splits_1 = list(kfold.split(csv_file_1))
        #splits_2 = list(kfold.split(csv_file_2))

        # labels = [x['label'] for x in csv_file]
        # KFold = StratifiedKFold(n_splits =5,random_state = seed, shuffle = True)
        # splits = list(KFold.split(csv_file, labels))

        train_idx_1, val_idx_1 = splits_1[fold_number]
        #print(train_idx_1)
        #train_idx_2, val_idx_2 = splits_2[fold_number]
        train_list = [csv_file_1[i] for i in train_idx_1]
        val_list = [csv_file_1[i] for i in val_idx_1]
        #train_list = [csv_file_1[i] for i in train_idx_1] + [csv_file_2[i] for i in train_idx_2] 
        #val_list = [csv_file_1[i] for i in val_idx_1] + [csv_file_2[i] for i in val_idx_2]
        
        train_dataset = bal_Dataset(image_path, train_list)
        val_dataset = bal_Dataset(image_path, val_list)
    
    # elif train_mode == 'tr_val':
    #
    #     train_dataset = bal_Dataset(fuyi_image_path, fuyi_csv_file)
    #     val_dataset = bal_Dataset(tongji_image_path, tongji_csv_file)

    
    # elif train_mode == '7_3':
    #
    #     if cv_data_source == 'fuyi':
    #         csv_file = fuyi_csv_file
    #         image_path = fuyi_image_path
    #     elif cv_data_source == 'tongji':
    #         csv_file = tongji_csv_file
    #         image_path = tongji_image_path
    #     # 按比例分
    #     # 获取每种标签的行号,并洗牌
    #     idx_0 = [i for i, x in enumerate(csv_file) if x['label'] == '0']
    #     idx_1 = [i for i, x in enumerate(csv_file) if x['label'] == '1']
    #     random.shuffle(idx_0)
    #     random.shuffle(idx_1)
    #     # 按比例划分数据集
    #     train_idx = idx_0[:int(0.7 * len(idx_0))] + idx_1[:int(0.7 * len(idx_1))]
    #     val_idx = idx_0[int(0.7 * len(idx_0)):] + idx_1[int(0.7 * len(idx_1)):]
    #
    #     train_list = [csv_file[i] for i in train_idx]
    #     val_list = [csv_file[i] for i in val_idx]
    #
    #     train_dataset = bal_Dataset(image_path, train_list)
    #     val_dataset = bal_Dataset(image_path, val_list)
        
    train_transforms = Compose(
            [#EnsureChannelFirst(),
            RandAffine(prob=0.5, translate_range=(4, 10, 10), padding_mode="border"),
            #RandFlip(prob=0.5),
            #RandRotate90(prob=0.5, spatial_axes=(0,1)),
            ToTensor()
            ])
    
    val_transforms = Compose(
            [#EnsureChannelFirst(),
             ToTensor()
            ])
    
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms
        
    return train_dataset,val_dataset


def get_dataset1(data_root, cv_data_source, train_mode, seed=42, fold_number=0):

    image_path1 = os.path.join(data_root, 'Dataset_five_N4_gamma')
    image_path2 = os.path.join(data_root, 'Dataset_five_N4_gamma_rotation')
    csv_path1 = os.path.join(data_root, 'Train_ISUP.csv')
    csv_path2 = os.path.join(data_root, 'Train_ISUP_rotation.csv')


    csv_file_1, csv_file_2 = [], []
    with open(csv_path1, encoding="gbk", mode="r") as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            csv_file_1.append(row)

    with open(csv_path2, encoding="gbk", mode="r") as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            csv_file_2.append(row)
            
    # Cross Validation
    if train_mode == 'Cross_Validation':

        kfold = KFold(10, shuffle=True, random_state=seed)
        splits_1 = list(kfold.split(csv_file_1))
        splits_2 = list(kfold.split(csv_file_2))
        train_idx_1, val_idx_1 = splits_1[fold_number]
        train_idx_2, val_idx_2 = splits_2[fold_number]
        
        train_list = [csv_file_1[i] for i in train_idx_1] + [csv_file_2[i] for i in train_idx_2] 
        val_list = [csv_file_1[i] for i in val_idx_1]
        
        train_dataset = bal_Dataset(image_path2, train_list)
        val_dataset = bal_Dataset(image_path1, val_list)

    train_transforms = Compose(
            [#EnsureChannelFirst(),
            RandAffine(prob=0.5, translate_range=(4, 10, 10), padding_mode="border"),
            #RandFlip(prob=0.5),
            #RandRotate90(prob=0.5, spatial_axes=(0,1)),
            ToTensor()
            ])
    
    val_transforms = Compose(
            [#EnsureChannelFirst(),
             ToTensor()
            ])
    
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms
        
    return train_dataset,val_dataset

# get_dataset('/nfs/wzy/CODE/classification/all/data/qlg')


# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.__next__() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()   # 训练集正负样本所占比例
    sampling_probabilities = relative_freq ** (-1)
    
    return sampling_probabilities

def modify_loader(loader, mode, ep=None, n_eps=None):
    class_count = np.unique(loader.dataset.dr, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[loader.dataset.dr]
    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    
    return mod_loader

def get_combo_loader(loader, base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling)

    balanced_loader = modify_loader(loader, mode='class')
    
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader


# the dataloader by ordinary augmentation
def get_ord_balanced_loader(loader, base_sampling='instance'):
    ord_balanced_loader = modify_loader(loader, mode='class')
    return ord_balanced_loader
