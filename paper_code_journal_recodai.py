from transformers import AutoImageProcessor, Dinov2Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import torchvision
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import copy
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from os import walk
import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score
import math

DATA_PATH = '../data_experiment_channel_1_label_5/'
MODEL_NAME = 'dinov2_vitl14'


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1),
                 padding=(1,1), dilation=(1,1), groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

# Function to replace conv layers with CDC layers
def replace_conv_with_cdc(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            cdc_layer = Conv2d_cd(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.bias is not None
            )
            setattr(module, name, cdc_layer)
        else:
            replace_conv_with_cdc(child)

class LithoVision(nn.Module):
    def __init__(self, dinov2_vits14, densenet):
        super(LithoVision, self).__init__()
        self.transformer = deepcopy(dinov2_vits14)
        self.densenet = deepcopy(densenet)
        self.fc1 = nn.Linear(2024, 768)
        self.fc2 = nn.Linear(768, 5)

    def forward(self,x):
        z = self.transformer(x)
        z = self.transformer.norm(z)

        y = self.densenet(x)

        concat = torch.cat((z,y), dim=1)

        
        features = torch.relu(self.fc1(concat))
        out = self.fc2(features)
        return out, features

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, num_features=2):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = num_features
        if torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss

def train_model(model, dataloaders, criterion, criterion_c, optimizer, optimizer_c, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10
    
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
       
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                optimizer_c.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                                     
                        outputs, features = model(inputs)
                        
                        center_loss = criterion_c(features, labels)
                        loss = criterion(outputs, labels)
                        loss =  center_loss*alpha + loss

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        for param in criterion_c.parameters():
                            param.grad.data *= (1./alpha)
                        optimizer_c.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:# and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history

def test_evaluation(model_ft, dataloaders, is_logits=False):
    y_pred = []
    y_true = []
    model_ft.eval()
    i = 0
    with torch.set_grad_enabled(False):
        for X, y in dataloaders['test']:
            X = X.to(device)
            pred, _ = model_ft(X)
            if is_logits:
                y_pred+= pred.cpu().numpy().tolist() ## pra pegar a distribuição e não o max               
            else:
                _, preds = torch.max(pred, 1)
                y_pred+= preds.cpu().numpy().tolist()
                
            y_true+= y.numpy().tolist()
            i+=1        
    return y_true, y_pred

if '__name__' == '__main__':
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)

    densenet = models.densenet121(pretrained=True)

    data_dir = DATA_PATH

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),#256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256), #256 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle = False if x=='test' else True, num_workers=4)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    class_names = image_datasets["train"].classes


    idx_to_class = {i:j for i, j in enumerate(class_names)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    # It's necessary calculate the weight, for simplicity we used a precomputed one.
    
    # count_by_class = [class_to_idx[i] for i in train.label.values]
    # cls_weight = compute_class_weight('balanced',classes=np.unique(count_by_class),y=count_by_class)
    cls_weight = np.array([0.95673077, 0.6958042 , 1.19161677, 1.18452381, 1.19879518])

    # Apply the replacement recursively
    replace_conv_with_cdc(densenet)

    model_ft = LithoVision(dinov2_vits14, densenet)

    model_ft = nn.DataParallel(model_ft, device_ids=[1,2])

    model_ft = model_ft.to(device)

    weights = torch.FloatTensor(cls_weight).to(device)
    #
    criterion = FocalLoss(weight=weights)

    criterion_c = CenterLoss(num_classes=7, num_features=768)

    criterion_c = criterion_c.to(device)

    alpha = 0.003

    # criterion = nn.CrossEntropyLoss(weight=weights)
        
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-6)

    optimizer_centloss = torch.optim.SGD(criterion_c.parameters(), lr = 1e-6)

    model_ft, hist_train, hist_val = train_model(model_ft, dataloaders, criterion, criterion_c, optimizer_ft, optimizer_centloss, num_epochs=20, is_inception=False)

    y_true, y_pred = test_evaluation(model_ft, dataloaders)

    print(f'B accuracy:{balanced_accuracy_score(y_true, y_pred)}',flush=True)
    print(f'F1:{f1_score(y_true, y_pred,average="weighted")}')
    print(f'Recall:{recall_score(y_true, y_pred, average="weighted")}')
    print(f'Precision:{precision_score(y_true, y_pred, average="weighted")}')