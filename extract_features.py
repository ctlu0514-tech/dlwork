# -*- coding: utf-8 -*-
import os

# 【新增】强制指定只使用 0 号显卡
# 这行代码要在 import torch 之前或者程序最开始设置最好，确保生效
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd  
from tqdm import tqdm

# 复用你原有的数据加载和模型构建代码
from dataloader_3dEPE import get_dataset
from models_EPE.get_model import get_arch

def extract_features_to_csv():
    # ================= 配置区域 =================
    model_name = 'resnet50'        # 必须和你训练时一致
    input_channels = 4             # 必须和你训练时一致
    attention = 'false'            # 必须和你训练时一致
    
    # 【请务必修改】为你实际找到的 best_model.pth 的完整路径
    weights_path = '/data/qh_20T_share_file/lct/CT67/dl_work/results/resnet50_20260109_005824_fold4_resnet50&invol_batch16_lr1e-06_epochs300/best_model.pth' 
    
    # 数据路径
    data_root = '/data/qh_20T_share_file/lct/CT67/dl_work/Dataset_crop'
    
    # 结果保存路径
    save_csv_path = '/data/qh_20T_share_file/lct/CT67/dl_work/extracted_features.csv' 
    # ===========================================

    # 1. 设置设备
    # 因为上面已经指定了 CUDA_VISIBLE_DEVICES='0'，所以这里的 'cuda' 就会自动对应 0 号卡
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"成功调用显卡: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告：未检测到显卡，正在使用 CPU (会很慢)")

    # 2. 加载模型架构
    print(f"正在创建模型: {model_name}...")
    model = get_arch(model_name=model_name, attention=attention, input_channels=input_channels)
    
    # 还原 DataParallel 结构以匹配权重文件的格式
    model = nn.DataParallel(model)
    
    # 3. 加载训练好的权重
    if os.path.exists(weights_path):
        print(f"正在加载权重: {weights_path}")
        # map_location 确保加载到当前指定的设备
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"错误：找不到权重文件，请检查路径是否正确 -> {weights_path}")

    # 4. 【关键】去掉分类头，只保留特征提取部分
    # ResNet50 最后一层叫 module.fc，我们将它替换为“直通车”(Identity)
    if hasattr(model.module, 'fc'):
        model.module.fc = nn.Identity()
        print("已移除分类层 (fc)，准备提取特征...")
    else:
        print("警告：未找到 fc 层，请确认模型结构！")

    model.to(device)
    model.eval() # 开启评估模式

    # 5. 准备数据 (这里以验证集为例，fold=0)
    # 如果你想提训练集特征，把 'Cross_Validation' 改一下或者把 val_dataset 换成 train_dataset
    _, val_dataset = get_dataset(data_root, 'fuyi', 'Cross_Validation', 42, 0) 
    
    data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # 6. 开始提取
    all_features = []
    all_labels = []

    print(f"开始提取特征，共有 {len(data_loader)} 张图片...")
    with torch.no_grad(): # 关掉梯度计算
        for batch in tqdm(data_loader):
            # 获取图片数据并放到设备上
            inputs = batch['all'].float().to(device)
            # 获取标签
            labels = batch['label'].item()
            
            # 提取特征
            features = model(inputs) 
            
            # 转成 numpy 数组并展平
            features_np = features.cpu().numpy().flatten()
            
            all_features.append(features_np)
            all_labels.append(labels)

    # 7. 保存为 CSV
    print("正在保存为 CSV 文件...")
    df = pd.DataFrame(all_features)
    df.insert(0, 'label', all_labels) # 把标签插在第一列
    
    df.to_csv(save_csv_path, index=False)
    print(f"成功！文件已保存至: {save_csv_path}")
    print(f"数据形状: {df.shape}")

if __name__ == '__main__':
    extract_features_to_csv()