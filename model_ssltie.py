from numpy.core import numeric
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('..')
# from networks import base_models

import ipdb
import random
import numpy as np

def normalize_img(value, vmax=None, vmin=None):
    #  pdb.set_trace()
    value1 = value.view(value.size(0), -1)
    value1 -= value1.min(1, keepdim=True)[0]
    value1 /= value1.max(1, keepdim=True)[0]
    return value1.view(value.size(0), value.size(1), value.size(2), value.size(3))

class AVENet_ssltie(nn.Module):

    def __init__(self, args):
        super(AVENet_ssltie, self).__init__()

        # -----------------------------------------------
        if args.heatmap_size == 28:
            from networks import base_models_hmap28 as base_models
        else:
            from networks import base_models as base_models
        self.imgnet = base_models.resnet18(modal='vision', pretrained=False)
        self.audnet = base_models.resnet18(modal='audio')
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.m = nn.Sigmoid()

        self.epsilon_temp = args.epsilon
        self.epsilon_temp2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg
        # self.random_threshold = args.random_threshold
        # self.soft_ep = args.soft_ep

        self.vision_fc1 = nn.Conv2d(1024,512 , kernel_size=(1, 1)) 
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(1, 1, 1)
        self.norm3 = nn.BatchNorm2d(1)
        self.vpool3 = nn.MaxPool2d(14, stride=14)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, audio, args, mode='val'):
        if mode != 'val':
            # Image
            B = image.shape[0]
            #self.mask = ( 1 -1 * torch.eye(B,B)).cuda()
            self.mask = ( 1 -100 * torch.eye(B,B)).cuda()
            img = self.imgnet(image)
            img = nn.functional.normalize(img, dim=1)

            # Audio
            aud = self.audnet(audio)
            aud = self.avgpool(aud).view(B,-1)
            aud = nn.functional.normalize(aud, dim=1)

            w = img.shape[-1]
            # img : B by Channel by w by h
            
            return img,aud
        if mode == 'val':
            self.epsilon =  args.epsilon
            self.epsilon2 = args.epsilon2
            # Image
            B = image.shape[0]
            self.mask = ( 1 -100 * torch.eye(B,B)).cuda()
            # import ipdb; ipdb.set_trace()
            img = self.imgnet(image)
            img = nn.functional.normalize(img, dim=1)
            # import ipdb; ipdb.set_trace()
            # Audio
            aud = self.audnet(audio)
            aud = self.avgpool(aud).view(B,-1)
            aud = nn.functional.normalize(aud, dim=1)
            # Join them
            out = torch.einsum('nchw,nc->nhw', img, aud).unsqueeze(1)   
            out1 = self.norm3(self.conv3(out))
            # out2 = self.vpool3(out1)
            A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)    
            A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])            
            A0_ref = self.avgpool(A0).view(B,B) # self.mask    BxB



            Pos = self.m((A - self.epsilon)/self.tau)      # positive region  mask
            if self.trimap:    
                Pos2 = self.m((A - self.epsilon2)/self.tau)   
                Neg = 1 - Pos2
            else:   
                Neg = 1 - Pos                               # negative region  mask 

            Pos_all =  self.m((A0 - self.epsilon)/self.tau)         
            A0_f = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) ) * self.mask   # easy negative BxB
            sim = A0_f 

            #  
            sim1 = (Pos * A).view(*A.shape[:2],-1).sum(-1) / (Pos.view(*Pos.shape[:2],-1).sum(-1))   #  positive    Bx1 
            sim2 = (Neg * A).view(*A.shape[:2],-1).sum(-1) / Neg.view(*Neg.shape[:2],-1).sum(-1)     #  hard negative Bx1

            if self.Neg:
                logits = torch.cat((sim1,sim,sim2),1)/0.07
            else:
                logits = torch.cat((sim1,sim),1)/0.07                                        # 0.07 is temperature           

            # generate one hot-labels
            # target = torch.Tensor([0 for _ in range(B) ]).to(torch.int64)
            # target = target.cuda(1, non_blocking=True )
            # labels = F.one_hot(target, num_classes=B+2)
            # labels = labels.to(torch.float32)
            # import ipdb; ipdb.set_trace()
            return A, logits, Pos, Neg, A0_ref
    def forward_ret(self, image, audio, args):
            # Image
            B = image.shape[0]
            #self.mask = ( 1 -1 * torch.eye(B,B)).cuda()
            self.mask = ( 1 -100 * torch.eye(B,B)).cuda()
            img = self.imgnet(image)
            img = nn.functional.normalize(img, dim=1)

            # Audio
            aud = self.audnet(audio)
            aud = self.avgpool(aud).view(B,-1)
            aud = nn.functional.normalize(aud, dim=1)

            w = img.shape[-1]
            # img : B by Channel by w by h
            
            return img,aud