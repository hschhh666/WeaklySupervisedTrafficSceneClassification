import argparse
import math
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.transforms.transforms import Resize
from util import Logger,print_running_time
from myModel import myResnet18


def set_model(args):
    # 加载模型
    model = myResnet18(args.feat_dim, parallel = False)
    
    # 加载预训练模型参数
    print("=> loading checkpoint '{}'".format(args.pretrained))
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    print('=> checkpoint epoch {}'.format(checkpoint['epoch']))

    # rename moco pre-trained keys
    state_dict = checkpoint['model']
    # linear_keyword = 'fc'
    for k in list(state_dict.keys()):
        state_dict[k.replace('module.','')] = state_dict[k]
        # retain only base_encoder up to before the embedding layer
        # if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
        #     # remove prefix
        #     state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # # delete renamed or unused k
        del state_dict[k]

    # 把预训练参数加载到模型中
    print("=> loaded pre-trained model '{}'".format(args.pretrained))
    msg = model.load_state_dict(state_dict, strict=1)
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
    model.cuda()
    model.eval()
    return model

def set_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True) # 不用打乱图片顺序，因为要看每张图的特征

    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=128, shuffle=False,num_workers=0, pin_memory=True)
    
    args.class_name = train_dataset.classes
    args.train_num = len(train_dataset)
    args.val_num = len(val_dataset)

    return train_loader, val_loader

# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
 
    # 自己修改forward函数
    def forward(self, x):
        for name, module in self.submodule._modules['model']._modules.items():
            if name == "fc": 
                x = x.view(x.size(0), -1)
                # return x
            x = module(x)
        return x

def get_feat(model, data_loader, args) -> np.array:
    feat_dim = args.feat_dim
    feature_extractor = FeatureExtractor(model)
    memory = torch.ones(args.n_data, feat_dim).cuda()
    targets = []
    batch_size = 0
    with torch.no_grad():
        for idx,(img, target) in enumerate(data_loader):
            batch_size = img.shape[0] if batch_size == 0 else img.shape[0]
            index = list(range(batch_size*idx, batch_size*idx + img.shape[0]))
            index = torch.tensor(index, dtype=torch.long).cuda()
            targets += list(target.numpy())
            img = img.cuda()
            feat = feature_extractor(img)
            memory.index_copy_(0,index,feat)
    memory = memory.cpu().numpy()
    return memory, targets


def vis_feat(feat, targets, save_name, args):
    reduced_feat_pca = PCA(n_components=2).fit_transform(feat)
    reduced_feat_tsne = TSNE(perplexity=100).fit_transform(feat)
    fig = plt.figure()
    fig.add_subplot(121)
    plt.scatter(reduced_feat_pca[:,0], reduced_feat_pca[:,1], c = targets, s = 1, alpha = 0.5)
    plt.title('Feature visualization by pca')
    fig.add_subplot(122)
    plt.scatter(reduced_feat_tsne[:,0], reduced_feat_tsne[:,1], c = targets, s = 1, alpha = 0.5)
    plt.title('Feature visualization by tsne')
    plt.savefig(os.path.join(args.result_path, save_name))


def process_feature(args): # 其实这个函数就相当于主程序了
    # 设置保存文件的路径
    args.result_path = ('/').join(((args.pretrained).split('/'))[:-1])
    args.result_path = (args.result_path).replace('modelPath','resultPath') 
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)

    # 获取特征
    # 配置模型
    model = set_model(args)

    # 配置DataLoader
    train_loader, val_loader = set_dataloader(args)

    args.n_data = args.train_num    
    print('Calculating train set feature...')
    feat, targets = get_feat(model, train_loader, args)
    # 保存特征
    np.save(os.path.join(args.result_path, 'train_feat.npy'), feat)
    np.save(os.path.join(args.result_path, 'train_targets.npy'), targets)
    # 可视化特征
    save_name = 'train_featvis.png'
    print('Visulization train set feature...')
    vis_feat(feat, targets, save_name, args)



    args.n_data = args.val_num    
    print('Calculating val set feature...')
    feat, targets = get_feat(model, val_loader, args)
    # 保存特征
    np.save(os.path.join(args.result_path, 'val_feat.npy'), feat)
    np.save(os.path.join(args.result_path, 'val_targets.npy'), targets)
    # 可视化特征
    save_name = 'val_featvis.png'
    print('Visulization val set feature...')
    vis_feat(feat, targets, save_name, args)



if __name__ == '__main__':
    # 其实就data和pretrained这俩参数需要输入，剩下的参数按默认值就行
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',help='path to dataset')
    parser.add_argument('--pretrained', default='', type=str,help='path to moco pretrained checkpoint')
    parser.add_argument('--num_class', default=4, type=int,help='Number of class.')
    parser.add_argument('--batch-size', default=128, type=int,metavar='N')
    parser.add_argument('--workers', default=6, type=int, metavar='N',help='number of data loading workers (default: 6)')
    parser.add_argument('--reduce_method', default='pca', type=str,help='feature dimention reduce method for visuilization, pca or tsne')

    args = parser.parse_args()
    args.feat_dim = 64
    args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/modelPath/20211119_17_31_19_lossMethod_softmax_NegNum_14_lr_0.03_decay_0.0001_bsz_8_featDim_64_/ckpt_epoch_120_Best.pth'
    args.data = '/home/hsc/Research/TrafficSceneClassification/data/HSD_masked_balanced'
    start = time.time()
    process_feature(args)
    print('Processing using time: %.0fs'%(time.time()-start))
    print('Done')