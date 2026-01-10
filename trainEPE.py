# -*- coding: utf-8 -*-
import os
# 设置一个新的临时文件夹路径（在你那个剩余 904G 的大盘里）
custom_temp_dir = '/data/qh_20T_share_file/lct/CT67/dl_work/tmp_cache'
# 如果文件夹不存在，自动创建
os.makedirs(custom_temp_dir, exist_ok=True)
# 强制告诉系统：把垃圾文件都写到这个大盘里，别去挤 /tmp 了
os.environ['TMPDIR'] = custom_temp_dir
import pathlib
from datetime import datetime
from tqdm import tqdm
from tqdm import trange
import argparse
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import torch.nn as nn
from dataloader_3dEPE import get_dataset, get_ord_balanced_loader
from utilsEPE import save_args, generate_result, loss_curve, save_confusion_matrix, save_result, sensitivityCalc, specificityCalc, writeROC, PR_curve
# from fusion import FeatureFusion, LearnedFeatureFusion, ProbabilityFusion
from models_EPE.get_model import get_arch
from sklearn import metrics as mts

from unbalanced_loss.focal_loss import FocalLoss
from unbalanced_loss.label_smoothing import LSR
from unbalanced_loss.unbalanced_loss import BinaryFocalLoss, GHMC_Loss, WBCEWithLogitLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--data_root', type=str,
                    default='/data/qh_20T_share_file/lct/CT67/dl_work/Dataset_crop', help='path to train dataset')
parser.add_argument('--train_save', type=str, default='TransFuse_S')
parser.add_argument('--devices', default='0', type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string') # 输入空的GPU的编号0-7 --device 0-7
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--seed', default=42, help="seed for train/val split")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--weights', type=str, default='best_model50.pth',
                        help='initial weights path')

parser.add_argument('--sampling', type=str, default='instance', help='sampling mode (instance, class, sqrt, progmultiprocessing)')
parser.add_argument('--loss', choices=['BinaryFocalLoss', 'GHMC_Loss', 'WBCEWithLogitLoss','CrossEntropyLoss','WCrossEntropyLoss','LabelSmoothing'], default='CrossEntropyLoss')
parser.add_argument('--model', choices=['MobileNet', 'MobileNetV3_Small', 'MobileNetV3_Large', 'ShuffleNet', 'DenseNet', 'resnet10', 'resnet18', 'resnet34','resnet50','resnet101','C3D', 'CNN_3D','Att_CNN_3D','LeNet', 'AlexNet','vit','resnext50','Resnext50','cotnet50','resnext101','vgg','SwinTransformer','Densenet36_fgpn'], default='resnet50')
parser.add_argument('--model_cla', choices=['MobileNet', 'resnet10', 'resnet18', 'resnet34','resnet50','resnet101','C3D', 'CNN_3D','Att_CNN_3D','LeNet', 'AlexNet','vit'], default='vit')
# parser.add_argument('--fusion_type', choices=['FeatureFusion', 'LearnedFeatureFusion', 'ProbabilityFusion', 'resnet_3d'], default='resnet_3d')
parser.add_argument('--attention', choices=['se','cbam', 'nam','AFF','false'], default='false')
parser.add_argument('--input_channels', default=4, type=int, help="input_channels (2,3,5)")

parser.add_argument('--invol', action='store_false')
parser.add_argument('--fusion_meta', action='store_true')
parser.add_argument('--cv_data_source', choices=['fuyi', 'tongji'], default='fuyi')
parser.add_argument('--train_mode', choices=['Cross_Validation', 'tr_val','7_3'], default='Cross_Validation')

def main(args):
    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{current_experiment_time}" \
                    f"_fold{args.fold if args.train_mode == 'Cross_Validation' else 'tr_fuyi&val_tongji'}" \
                    f"_{args.model}{'&invol' if args.invol else ''}{'&fusion_meta' if args.fusion_meta else ''}" \
                    f"_batch{args.batch_size}" \
                    f"_lr{args.lr}_epochs{args.epoch}"
    
    # args.save_folder = pathlib.Path(f"/nfs/wzy/CODE/pse/result/good/fuyi/8.18/resnet34/{args.exp_name}") # 存结果
    args.save_folder = pathlib.Path(f"./results/{args.model}_{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    save_args(args)
    
    train_dataset, val_dataset = get_dataset(args.data_root, args.cv_data_source, args.train_mode, args.seed, args.fold)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,    batch_size=2, shuffle=True,
        num_workers=args.workers, pin_memory=False,drop_last=True)
    '''
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=False)
    '''
    batch_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=args.workers, pin_memory=False)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    
    def ret_class_count(loader):
    #0和1的个数
        class_count = np.unique(loader.dataset.dr, return_counts=True)[1]
        return class_count
    train_class_count = ret_class_count(train_loader)
    #计算每个类别的权重
    train_class_weights = 1. / torch.tensor(train_class_count, dtype=torch.float)
    print("train_class_weights: ", train_class_weights)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # if args.fusion_type == 'FeatureFusion':
    #     model = FeatureFusion(meta_features = 4, model_name = args.model)
    # elif args.fusion_type == 'LearnedFeatureFusion':
    #     model = LearnedFeatureFusion(meta_features = 4, model_name = args.model)
    # elif args.fusion_type == 'ProbabilityFusion':
    #     model = ProbabilityFusion(meta_features = 4, model_name = args.model)
    # elif args.fusion_type == 'resnet_3d':
    
    model = get_arch(model_name=args.model, attention=args.attention, input_channels=args.input_channels)
    model = nn.DataParallel(model)
    model_cla = get_arch(model_name=args.model_cla, attention=args.attention, input_channels=args.input_channels)
    
    #seed=42
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    
    if args.loss == 'BinaryFocalLoss': loss_function = BinaryFocalLoss()
    elif args.loss == 'GHMC_Loss': loss_function = GHMC_Loss(alpha = 0.25, bins = 25)
    elif args.loss == 'WBCEWithLogitLoss': loss_function = WBCEWithLogitLoss() 
    elif args.loss == 'CrossEntropyLoss':loss_function = torch.nn.CrossEntropyLoss()
    elif args.loss == 'WCrossEntropyLoss':
        loss_function = torch.nn.CrossEntropyLoss(weight=train_class_weights.cuda())
    elif args.loss == 'LabelSmoothing':
        loss_function = LSR()
        
    # # load pretrain weights
    # model_weight_path = "/nfs/wzy/CODE/pse/50/pre-training/best_model50.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # pretrained_dict = torch.load(model_weight_path,map_location='cpu')
    # model_dict=model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'conv1' not in k)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict,strict=False)
    
    model.cuda()
    if args.invol:
        model_cla = model_cla
        model_cla.cuda()
    else:
        model_cla = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, eps=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scheduler = None
    if model_cla is not None:
        optimizer_cla = torch.optim.Adam(model_cla.parameters(), lr=args.lr, weight_decay=0, eps=1e-4)
        # scheduler_cla = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cla, 'min')
        scheduler_cla = None
    print(f"000000model_cla is: {model_cla}")
    print(f"000000invol is: {args.invol}")
    print(f"000000fusion_meta is: {args.fusion_meta}")
    best_loss = np.inf
    best_acc = 0.0
    save_path = os.path.join(args.save_folder, 'best_model.pth')
    train_steps = len(train_loader) + 1
    train_loss_curve, val_loss_curve = [], []
    
    ord_balanced_loader = get_ord_balanced_loader(train_loader, base_sampling=args.sampling)
    train_loader = ord_balanced_loader
    
    for epoch in range(args.epoch):
        
        if model_cla is not None:
            model_cla.train()
            _, _, _, _, _  = step(args, train_loader, model_cla, loss_function, optimizer_cla, epoch, args.batch_size,
                                                 train_steps, scheduler_cla, image_type='all', task='label',
                                                 fusion_meta = False)
            
        model.train()
        train_loss, train_acc, labels_list, results_list, outputs_list  = step(args, train_loader, model, loss_function, optimizer, epoch, args.batch_size, train_steps, 
                                   scheduler, image_type='all', task='label', model_cla=model_cla,
                                  save_folder=args.save_folder, fusion_meta = args.fusion_meta)

        # validate
        val_steps = len(val_loader)
        model.eval()
        val_loss, val_acc, _, _, _ = step(args, val_loader, model, loss_function, optimizer, epoch, 2, val_steps,
                          model_cla=model_cla, fusion_meta = args.fusion_meta)
        
        
        print('[epoch %d] train_loss: %.3f   var_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' % 
              (epoch + 1, train_loss,val_loss, train_acc, val_acc))
        
        with open(f"{args.save_folder}/val.txt", mode="a") as f:
                print(f"[epoch {epoch + 1}] train_loss: {train_loss} val_loss: {val_loss}  train_accuracy: {train_acc} val_accuracy: {val_acc} ",
                    file=f)
                
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
        
        # scheduler.step(val_loss) # 动态学习率
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
    
    measure(labels_list, results_list, outputs_list, args)
    loss_curve(train_loss_curve, val_loss_curve, args.save_folder, args.epoch)
    print('Finished Training')

    try:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint)
        global fin_result
        result = generate_result(batch_loader, model, args)
        if arguments.fold == 0:
            fin_result = result
        else:
            for key, value in result.items():
                fin_result[key] += value
            if arguments.fold == 9:
                average_dict = {key: value / 10 for key, value in fin_result.items()}
                save_average_csv = os.path.join(args.save_folder, "average_result.csv")
                with open(save_average_csv, mode='a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.DictWriter(f,
                                                fieldnames=['accuracy', 'recall', 'f1_score', 'kappa_score', 'sensitivity',
                                                            'specificity', 'PPV_score', 'NPV_score', 'auc'])
                    if os.path.getsize(save_average_csv) == 0:
                        csv_writer.writeheader()
                    csv_writer.writerow(average_dict)

    except KeyboardInterrupt:
        print("Stopping right now!")
        
def step(args, data_loader, model, loss_function, optimizer, epoch, batchsize, steps,
         scheduler=None, image_type='all',task="label", model_cla=None, save_folder=None, fusion_meta=False ):
    _loss = 0.0
    _acc = 0.0
    labels_list, outputs_list, results_list = [], [], []
    num_samples_1, num_samples_0 = 0, 0
    
    mode = "train" if model.training else "val"
    with trange(len(data_loader)) as t:
        for i, batch in enumerate(data_loader):
            
            inputs = batch[image_type].float().cuda()

            labels = batch[task].cuda()
            #age = batch['age'].cuda()
            #height = batch['height'].cuda()
            #weight = batch['weight'].cuda()
            #tpsa1 = batch['tpsa1'].cuda()
            #tpsa2 = batch['tpsa2'].cuda()
            #meta = torch.cat((age.unsqueeze(1), height.unsqueeze(1), weight.unsqueeze(1), tpsa1.unsqueeze(1), tpsa2.unsqueeze(1)), dim=1)
            
            
            #if fusion_meta:
            #    outputs, _ = model(inputs, meta=meta)
            #else:
            #   outputs, _ = model(inputs)
            
            # meta = batch['metadata'].cuda()
            #
            # if model_cla is not None:
            #     if fusion_meta:
            #         _, feature = model_cla(inputs)
            #         outputs, _ = model(inputs, feature=feature, meta=meta)
            #     else:
            #         _, feature = model_cla(inputs)
            #         outputs, _ = model(inputs, feature=feature)
            # else:
            #     if fusion_meta:
            #         outputs, _ = model(inputs, meta=meta)
            #     else:
            #         outputs, _ = model(inputs)

            #outputs, _ = model(inputs)
            outputs= model(inputs)

            predict = torch.max(outputs, dim=1)[1]
            _acc += torch.eq(predict, labels).sum().item()
            loss = loss_function(outputs, labels.long())
            
            # compute gradient
            if model.training:
                optimizer.zero_grad()
                loss = loss.requires_grad_(True)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(loss)

            _loss += loss.item()
            labels_list.append(labels.cpu().numpy()[0])
            results_list.append(predict.cpu().numpy()[0])
            outputs_list.append(outputs[0][1].item())
            num_samples_1 += torch.sum(labels == 1).item()
            num_samples_0 += torch.sum(labels == 0).item()
            t.update()

        _acc = _acc / (steps * batchsize)
        _loss = _loss / steps
        t.set_postfix(loss=_loss)
        tqdm.write(f"num_samples_1 = {num_samples_1}, num_samples_0 = {num_samples_0}")
        return _loss, _acc, labels_list, results_list, outputs_list


def measure(labels, results, outputs , args):
    train_folder = args.save_folder / "train"
    train_folder.mkdir(parents=True, exist_ok=True)
    cmatrix = mts.confusion_matrix(labels, results)
    if cmatrix.shape == (2, 2):
        # 二分类情况
        TN, FP, FN, TP = cmatrix.ravel() # 展平获取
    else:
        # 原始代码 (针对特定多分类任务)
        FP = cmatrix[0][1]
        FN = cmatrix[0][2]
        TP = cmatrix[1][1]
        TN = cmatrix[1][2]
    acc = mts.accuracy_score(labels, results)
    recall = mts.recall_score(labels, results, average='macro')
    PPV = TP/(TP+FP+1e-6)
    NPV = TN/(TN+FN+1e-6)
    
    f1_score = mts.f1_score(labels, results, average='macro')
    kappa_score = mts.cohen_kappa_score(labels, results)
    
    sensitivity = sensitivityCalc(labels, results)
    specificity = specificityCalc(labels, results)

    fin_result = dict(
        accuracy=acc, recall=recall, f1_score=f1_score, kappa_score=kappa_score,
        sensitivity=sensitivity, specificity=specificity, PPV_score = PPV, NPV_score = NPV
    )
    save_csv = os.path.join(train_folder, f"result_fold{args.fold}.csv")
    # 使用 with 语句，自动处理关闭和保存
    with open(save_csv, mode='a', encoding='utf-8', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['accuracy', 'recall', 'f1_score', 'kappa_score', 'sensitivity',
                                                   'specificity','PPV_score','NPV_score'])
        # 注意：如果是追加模式('a')，一直写header会导致csv里有很多重复表头。
        # 建议判断文件是否为空再写header，或者干脆手动管理。
        # 但为了最小改动，先保持原样，只是加上 with
        if os.path.getsize(save_csv) == 0: # 只有文件为空时才写表头
             csv_writer.writeheader()
        csv_writer.writerow(fin_result)
    print('accuracy: %.3f  recall: %.3f  f1_score: %.3f kappa_score: %.3f sensitivity: %.3f specificity: %.3f PPV: %.3f NPV: %.3f' %
          (acc, recall, f1_score, kappa_score, sensitivity, specificity,PPV,NPV))
    
    save_confusion_matrix(cmatrix, ['0', '1'], f'{train_folder}', args.fold)
    
    # writeROC(outputs, labels, f'{args.save_folder}', args.fold)
    PR_curve(outputs,labels,f'{train_folder}', args.fold)

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    fin_result = {}
    if arguments.train_mode == 'Cross_Validation':
        for i in range(10):
            arguments.fold = i
            main(arguments)
    else:
        main(arguments)
        
        

