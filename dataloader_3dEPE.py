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
def extract_file_names(serial, folder_path):
    # 安全检查：如果文件夹都不存在，直接返回None
    if not os.path.exists(folder_path):
        return None
    
    for file_name in os.listdir(folder_path):
        # 只要文件名包含序列名 (如 'CT') 且以 .nii.gz 结尾
        if serial in file_name and file_name.endswith('.nii.gz'):
            return file_name
    return None
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
    def __init__(self, image_root, csv_file, transforms=None, task='label', is_train=True):
        self.root = image_root
        self.datas = []
        self.dr = []
        self.transforms = transforms
        
        # 1. 设定我们要读取的 4 个模态 (去掉了 PET，只保留 CT 和 MRI)
        # 如果你想用 PET，可以在列表里加上 'PET'
        Serial = ['ADC', 'DWI', 'T2', 'CT']
        
        # 2. 定义两个数据源文件夹的绝对路径
        # image_root 就是你传入的 /nfs/zc1/qianliexian/dataset/
        mri_dir = os.path.join(self.root, 'mpMri_nii')
        petct_dir = os.path.join(self.root, 'PETCT_nii')

        for row in csv_file:
            patient_id = str(row['id']) # 确保转为字符串
            patient_label = row['isup2'] # 确保这里是你 CSV 里的标签列名
            
            images = []
            missing_info = "" # 记录缺了啥
            
            for serial in Serial:
                # 3. 核心分流逻辑：CT 去 PETCT_nii 找，其他的去 mpMri_nii 找
                if serial == 'CT' or serial == 'PET':
                    current_root = petct_dir
                else:
                    current_root = mri_dir
                
                # 拼接病人文件夹: .../PETCT_nii/1002
                patient_folder = os.path.join(current_root, patient_id)
                
                # 调用上面的函数查找文件
                niiname = extract_file_names(serial, patient_folder)
                
                if niiname is None:
                    # 如果找不到，记录下来方便调试
                    missing_info = f"{serial} in {patient_folder}"
                    break # 只要缺一个，这个病人就不能用了，跳出循环

                image_file = os.path.join(patient_folder, niiname)
                
                # 读取图像
                try:
                    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_file))).astype(np.float32)
                    images.append(image)
                except:
                    missing_info = f"ReadError {image_file}"
                    break

            # 4. 只有集齐 4 个模态才加入训练列表
            if len(images) == 4:
                if task =='label':
                    label = float(patient_label)
                
                self.dr.append(int(label))
                patient = dict(id=patient_id, images=images, label=label)
                self.datas.append(patient)
            else:
                # 打印被跳过的病人 (前几次打印，防止刷屏)
                # 这样你就能看到到底是因为缺文件，还是路径拼错了
                print(f"Skipping {patient_id}: Missing {missing_info}")

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
        img_normalize = img_normalize.astype(np.float32)
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


# 请确保文件最上面有这一行: import pandas as pd

def get_dataset(data_root, cv_data_source, train_mode, seed=42, fold_number=0):
    # --- 1. 基础路径配置 ---
    image_path = data_root 
    csv_path = os.path.join(data_root, 'qianliexian_clinical_isup.csv')
    
    # 这是我们之前生成的固定划分名单
    split_csv_path = './dataset_split.csv' 
    
    print(f"正在加载数据: {csv_path}")
    
    # --- 2. 读取划分名单 (关键修改) ---
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError("报错: 找不到 dataset_split.csv！请先运行 make_split.py 生成该文件。")
    
    # 读取 split 文件并转为字典 {id: fold}，方便快速查找
    # (将 ID 转为 string 格式，防止与 csv 读取的类型不一致)
    split_df = pd.read_csv(split_csv_path)
    id_to_fold = dict(zip(split_df['id'].astype(str), split_df['fold']))
    
    # --- 3. 读取主 CSV 并根据名单分配 ---
    train_list = []
    val_list = []
    
    # 使用 utf-8-sig 防止 Windows CSV 的 BOM 乱码
    with open(csv_path, encoding="utf-8-sig", mode="r") as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            pat_id = str(row['id']) # 获取当前病人的 ID
            
            # 检查这个病人在不在我们的名单里
            if pat_id in id_to_fold:
                pat_fold = id_to_fold[pat_id]
                
                # [核心逻辑] 
                # 如果这个病人的 fold 编号 == 我们当前要跑的 fold_number，他是验证集
                # 否则，他是训练集
                if pat_fold == fold_number:
                    val_list.append(row)
                else:
                    train_list.append(row)
            else:
                # 如果名单里没这个号(可能是被清洗掉的数据)，就跳过
                pass

    print(f"Fold {fold_number} 加载完毕: 训练集 {len(train_list)} 例, 验证集 {len(val_list)} 例")
    
    # --- 4. 创建 Dataset 对象 (保持原样) ---
    train_dataset = bal_Dataset(image_path, train_list)
    val_dataset = bal_Dataset(image_path, val_list)
    
    # --- 5. 数据增强 (保持你原来的设置) ---
    train_transforms = Compose([
            # 你的原始增强逻辑
            RandAffine(prob=0.5, translate_range=(4, 10, 10), padding_mode="border"),
            ToTensor()
            ])
    
    val_transforms = Compose([
             ToTensor()
            ])
    
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms
        
    return train_dataset, val_dataset


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
