import os
import pprint

import torch
import yaml
import csv
import numpy as np
# from torch.cuda.amp import autocast
from sklearn import metrics as mts
import matplotlib
matplotlib.use('AGG')
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
global PCAfeature_val
global PCAfeature_val_name
PCAfeature_val = []
PCAfeature_val_name = []
from openpyxl import Workbook
import re

# def PCA_feature(feature_list,namelist,save_path):   #PCA特征降维 并输出投影后的方差分布图像
   
#     feature = np.array(feature_list)
#     print(feature.shape)
#     pca1 = PCA(n_components=15)
#     pca1.fit(feature)
#     #返回所保留的n个成分各自的方差百分比
#     print(pca1.explained_variance_ratio_)
#     print(pca1.explained_variance_)
#     feature = pca1.transform(feature) #降维后的feature
#     print(feature.shape)
#     print("成功输出五折的PCA")
#     finallist = []
#     num = len(namelist)
#     for i in range(0,num):
#         a = feature[i]
#         a = np.insert(a,0,namelist[i][0])
#         a = np.insert(a,0,namelist[i][1])
#         finallist.append(a)
#     save_path1= os.path.join(save_path, f"PCA_final.csv")
#     np.savetxt(save_path1,finallist,delimiter=', ', fmt='%f')


def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


def generate_result(data_loader, model, args):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    acc = 0.
    feature_list = []
    #serial = args.serial
    classes = ['0 ','1']
    cmatrix = np.zeros(shape=(4, 4), dtype=int)
    labels, outputs, results, metrics_list= [], [], [], []
    # global PCAfeature_val
    # global PCAfeature_val_name
    patient_id_list = []

    for i, batch in enumerate(data_loader):
        # measure data loading time
        images = batch["all"].cuda()
        patient_id = batch["patient_id"]
        label = batch["label"]
        #age = batch['age'].cuda()
        #height = batch['height'].cuda()
        #weight = batch['weight'].cuda()
        #tpsa1 = batch['tpsa1'].cuda()
        #tpsa2 = batch['tpsa2'].cuda()
        #meta = torch.cat((age.unsqueeze(1), height.unsqueeze(1), weight.unsqueeze(1), tpsa1.unsqueeze(1), tpsa2.unsqueeze(1)), dim=1)
        # meta = batch['metadata'].cuda()

        
        with torch.no_grad():
            #output, _ = model(images)
            output = model(images)
           #output, _ = model(images,meta=meta)
            # feature1 = feature1.cpu()
            # feature1 = feature1.numpy()
            # feature1 = feature1.flatten()
            predict = torch.max(output, dim=1)[1]
            # feature2 = feature2.cpu()
            # feature2 = feature2.numpy()
            # feature2 = feature2.flatten()
            
        # feature_list.append(feature)
        
        label = label.numpy()
        predict = predict.cpu()
        predict = predict.numpy()
        
        metrics = dict(
            patient_id=patient_id[0], label=label[0], predict=predict[0], outputs=output[0][1].item()
        )
        labels.append(label[0])
        results.append(predict[0])
        outputs.append(output[0][1].item())
        metrics_list.append(metrics)
        patient_id_list.append(patient_id)
        # if(on == 'val'):
        #     PCAfeature_val.append(feature2)
        #     PCAfeature_val_name.append([label[0],patient_id[0]])
        # feature1 = np.insert(feature1,0,label[0])
        # feature1 = np.insert(feature1,0,patient_id[0])
        # feature.append(feature1)
        # feature2 = np.insert(feature2,0,label[0])
        # feature2 = np.insert(feature2,0,patient_id[0])
        # bigfeature.append(feature2)
    cmatrix = mts.confusion_matrix(labels, results)
    FP = cmatrix[0][1]
    FN = cmatrix[1][0]
    TP = cmatrix[1][1]
    TN = cmatrix[0][0]

    acc = mts.accuracy_score(labels, results)
    recall = mts.recall_score(labels, results, average='macro')
    PPV = TP/(TP+FP+1e-6)
    NPV = TN/(TN+FN+1e-6)
    
    f1_score = mts.f1_score(labels, results, average='macro')
    kappa_score = mts.cohen_kappa_score(labels, results)
    
    sensitivity = sensitivityCalc(labels, results)
    specificity = specificityCalc(labels, results)
    auc = writeROC(outputs, labels, f'{args.save_folder}', args.fold, classes)
    fin_result = dict(
        accuracy=acc, recall=recall, f1_score=f1_score, kappa_score=kappa_score,
        sensitivity=sensitivity, specificity=specificity, PPV_score = PPV, NPV_score = NPV,auc=auc
    )
    print('accuracy: %.3f  recall: %.3f  f1_score: %.3f kappa_score: %.3f sensitivity: %.3f specificity: %.3f PPV: %.3f NPV: %.3f' %
          (acc, recall, f1_score, kappa_score, sensitivity, specificity,PPV,NPV))
    
    # save_feature(feature_list,patient_id_list,args.save_folder,args)
    
    save_result(metrics_list, fin_result, f'{args.save_folder}', args.fold)
    save_confusion_matrix(cmatrix, classes, f'{args.save_folder}', args.fold)

   
    # save_middle_feature(feature , bigfeature , f'{args.save_folder}/{on}', args.fold)
    # if(args.fold == 4 and on == 'val'):
    #     PCA_feature(PCAfeature_val,PCAfeature_val_name,f'{args.save_folder}/{on}')
    PR_curve(outputs,labels,f'{args.save_folder}', args.fold)
    return fin_result
    



def save_confusion_matrix(confusion_matrix, classes, save_path, fold):
    plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar()

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            plt.annotate(confusion_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    indices = range(len(confusion_matrix))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('confusion matrix')

    save_png = os.path.join(save_path, f"confusion_matrix_fold{fold}.png")
    save_npy = os.path.join(save_path, f"confusion_matrix_fold{fold}.csv")

    plt.savefig(save_png)
    np.savetxt(save_npy, confusion_matrix)


def save_result(metrics_list, fin_result, save_path, fold):
    save_csv = os.path.join(save_path, f"patients_result_fold{fold}.csv")
    f = open(save_csv, mode='a', encoding='utf-8', newline='')

    csv_writer = csv.DictWriter(f, fieldnames=['patient_id', 'image_id', 'label', 'predict', 'outputs'])  # 列名
    csv_writer.writeheader()

    for item in range(len(metrics_list)):
        csv_writer.writerow(metrics_list[item])

    save_csv = os.path.join(save_path, f"result_fold{fold}.csv")
    f = open(save_csv, mode='a', encoding='utf-8', newline='')

    csv_writer = csv.DictWriter(f, fieldnames=['accuracy', 'recall', 'f1_score', 'kappa_score', 'sensitivity',
                                               'specificity','PPV_score','NPV_score','auc'])
    csv_writer.writeheader()
    csv_writer.writerow(fin_result)


def sensitivityCalc(Labels,Predictions):
    cm1 = mts.confusion_matrix(Labels, Predictions)
    TN = cm1[0, 0]
    FP = cm1[0, 1]
    FN = cm1[1, 0]
    TP = cm1[1, 1]
    sensitivity = TP / (TP + FN) + 1e-6
    return sensitivity


def specificityCalc(Labels, Predictions):
    cm1 = mts.confusion_matrix(Labels, Predictions)
    TN = cm1[0, 0]
    FP = cm1[0, 1]
    FN = cm1[1, 0]
    TP = cm1[1, 1]
    specificity = TN / (FP + TN) + 1e-6
    return specificity


def writeROC(outputs, labels, save_folder, fold,classes):

    fpr, tpr, thresholds = roc_curve(labels, outputs)
    roc_auc = auc(fpr, tpr)

    # #约登指数输出最佳阈值
    # youden_idx = tpr-fpr
    # best_threshold_idx = np.argmax(youden_idx)
    # best_threshold = thresholds[best_threshold_idx]
    # outputs = np.where(outputs>=best_threshold,1,0)
    # cm = mts.confusion_matrix(labels,outputs)
    
    # plt.figure()
    # plt.imshow(cm, cmap=plt.cm.Blues)
    # plt.colorbar()

    # for i in range(len(cm)):
    #     for j in range(len(cm)):
    #         plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # indices = range(len(cm))
    # plt.xticks(indices, classes)
    # plt.yticks(indices, classes)
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.title('confusion matrix')

    # save_png = os.path.join(save_folder, f"youden_confusion_matrix_fold.png")

    # plt.savefig(save_png)
    # plt.close()
#

    plt.figure()
    
    plt.plot(fpr, tpr,
            label=' ROC curve (area = {0:0.2f})'
                ''.format(roc_auc),
            color='b', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_folder, f"ROC_fold{fold}.png"))
    return roc_auc

# def save_middle_feature(feature_numpy,bigfeature,save_path, fold):
#     save_path1= os.path.join(save_path, f"middle_feature{fold}.csv")
#     np.savetxt(save_path1,feature_numpy,delimiter=', ', fmt='%f')
#     save_path2= os.path.join(save_path, f"middle_bigfeature{fold}.csv")
#     np.savetxt(save_path2,bigfeature,delimiter=', ', fmt='%f')
    
def PR_curve(outputs,labels,save_folder, fold):
    precision, recall, _ = precision_recall_curve(labels, outputs)
    plt.figure(figsize=(12,8))
    AP = average_precision_score(labels, outputs)
    print('Average precision-recall score: {0:0.2f}'.format(AP))
    plt.title('2-class Precision-Recall curve: Average precision-recall score={0:0.2f}'.format(AP))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(recall,precision)
    # plt.show()
    save_png = os.path.join(save_folder, f"PR_fold{fold}.png")
    plt.savefig(save_png)
    


def save_feature(feature_list, patient_id_list, save_path, args):
    workbook = Workbook()
    save_file = os.path.join(save_path,f"patient_feature_fold{args.fold}.xlsx")
    worksheet = workbook.active

    worksheet.title = "Sheet1"
    x, y, z = np.array(feature_list).shape
    print(x)
    print(len(patient_id_list))
    for i in range(x):
        worksheet.append(list(np.insert((feature_list[i][0]), 0, ["".join(re.findall("\d+", str(patient_id_list[i])))])))
    workbook.save(filename=save_file)


def loss_curve(train_losses, val_losses, save_folder, epoch):
    diff_len = len(train_losses) - len(val_losses)
    val_losses_padded = np.pad(val_losses, (0, diff_len), 'constant', constant_values=np.nan)
    plt.figure()
    plt.plot(range(epoch), train_losses, label='Training Loss')
    plt.plot(range(epoch), val_losses_padded, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_png = os.path.join(save_folder, f"loss_curve.png")
    plt.savefig(save_png)
    