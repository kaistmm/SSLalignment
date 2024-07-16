import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class EZVSL(nn.Module):
    def __init__(self, tau, dim):
        super(EZVSL, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
        return loss, Slogits

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_proj(img)
        img = nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio)
        aud = self.aud_proj(aud)
        aud = nn.functional.normalize(aud, dim=1)

        # Compute loss
        loss, logits = self.max_xmil_loss(img, aud)

        # Compute avl maps
        with torch.no_grad():
            B = img.shape[0]
            Savl = logits[torch.arange(B), torch.arange(B)]

        return loss, Savl
    
class SLAVC(nn.Module):
    def __init__(self, tau, dim, dropout_img, dropout_aud, momentum_img, momentum_aud, use_mom_eval, num_neg=None):
        super(SLAVC, self).__init__()
        self.tau = tau
        self.num_neg = num_neg

        # Vision model
        self.imgnet = self.build_imgnet()
        self.img_dropout = nn.Dropout(p=dropout_img)
        self.img_proj1 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.img_proj2 = nn.Conv2d(512, dim, kernel_size=(1, 1))

        # Audio model
        self.audnet = self.build_audnet()
        self.aud_proj1 = nn.Linear(512, dim)
        self.aud_proj2 = nn.Linear(512, dim)
        self.aud_dropout = nn.Dropout(p=dropout_aud)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj1, self.aud_proj1, self.img_proj2, self.aud_proj2]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

        # momentum vision & audio models
        self.momentum_imgnet = self.build_imgnet()
        self.momentum_img_proj1 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.momentum_img_proj2 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.momentum_audnet = self.build_audnet()
        self.momentum_aud_proj1 = nn.Linear(512, dim)
        self.momentum_aud_proj2 = nn.Linear(512, dim)

        self.m_img = momentum_img
        self.m_aud = momentum_aud
        self.use_mom_eval = use_mom_eval

        # initialize momentum_encoders
        self.initialize_momentum_encoder(self.imgnet, self.momentum_imgnet)
        self.initialize_momentum_encoder(self.img_proj1, self.momentum_img_proj1)
        self.initialize_momentum_encoder(self.img_proj2, self.momentum_img_proj2)
        self.initialize_momentum_encoder(self.audnet, self.momentum_audnet)
        self.initialize_momentum_encoder(self.aud_proj1, self.momentum_aud_proj1)
        self.initialize_momentum_encoder(self.aud_proj2, self.momentum_aud_proj2)

    @torch.no_grad()
    def initialize_momentum_encoder(self, base_encoder, momentum_encoder):
        for param_b, param_m in zip(base_encoder.parameters(), momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_encoder(self, m, base_encoder, momentum_encoder):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(base_encoder.parameters(), momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def build_imgnet(self):
        imgnet = resnet18(pretrained=True)
        imgnet.avgpool = nn.Identity()
        imgnet.fc = nn.Identity()
        return imgnet

    def build_audnet(self):
        audnet = resnet18()
        audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        audnet.fc = nn.Identity()
        return audnet

    def forward_img_features(self, imgnet, improj1, improj2, image):
        # Image
        img = imgnet(image).unflatten(1, (512, 7, 7))#14,14))
        img = self.img_dropout(img)
        img1 = improj1(img)
        img1 = nn.functional.normalize(img1, dim=1)
        img2 = improj2(img)
        img2 = nn.functional.normalize(img2, dim=1)
        return img1, img2, img

    def forward_aud_features(self, audnet, audproj1, audproj2, audio):
        # Audio
        aud = audnet(audio)
        aud = self.aud_dropout(aud)
        aud1 = audproj1(aud)
        aud1 = nn.functional.normalize(aud1, dim=1)
        aud2 = audproj2(aud)
        aud2 = nn.functional.normalize(aud2, dim=1)
        return aud1, aud2, aud

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        if img.ndim == 4 and aud.ndim == 2:
            Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
            labels = torch.arange(B).long().to(img.device)
        elif img.ndim == 5 and aud.ndim == 2:
            Slogits = torch.einsum('nmchw,nc->nmhw', img, aud) / self.tau
            labels = torch.zeros(B).long().to(img.device)
        elif img.ndim == 4 and aud.ndim == 3:
            Slogits = torch.einsum('nchw,nmc->nmhw', img, aud) / self.tau
            labels = torch.zeros(B).long().to(img.device)
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, image, audio, mode='train'):
        # compute features
        img1, img2,img = self.forward_img_features(self.imgnet, self.img_proj1, self.img_proj2, image)
        aud1, aud2,aud = self.forward_aud_features(self.audnet, self.aud_proj1, self.aud_proj2, audio)

        with torch.no_grad():  # no gradient
            if mode == 'train':
                self._update_momentum_encoder(self.m_img, self.imgnet, self.momentum_imgnet)  # update the vision momentum encoder
                self._update_momentum_encoder(self.m_img, self.img_proj1, self.momentum_img_proj1)  # update the vision momentum projection
                self._update_momentum_encoder(self.m_img, self.img_proj2, self.momentum_img_proj2)  # update the vision momentum projection
                self._update_momentum_encoder(self.m_aud, self.audnet, self.momentum_audnet)  # update the audio momentum encoder
                self._update_momentum_encoder(self.m_aud, self.aud_proj1, self.momentum_aud_proj1)  # update the audio momentum projection
                self._update_momentum_encoder(self.m_aud, self.aud_proj2, self.momentum_aud_proj2)  # update the audio momentum projection

            # compute momentum features as targets
            img1_trg, img2_trg,img_trg = self.forward_img_features(self.momentum_imgnet, self.momentum_img_proj1, self.momentum_img_proj2, image)
            aud1_trg, aud2_trg,aud_trg = self.forward_aud_features(self.momentum_audnet, self.momentum_aud_proj1, self.momentum_aud_proj2, audio)

        # Compute loss
        i2a_1 = F.softmax(torch.einsum('nchw,mc->nmhw', img1, aud1_trg).flatten(-2, -1) / self.tau, dim=1)
        i2a_2 = F.softmax(torch.einsum('nchw,mc->nmhw', img2, aud2_trg).flatten(-2, -1) / self.tau, dim=2)
        i2a = torch.log((i2a_1 * i2a_2).sum(2))    # nm

        a2i_1 = F.softmax(torch.einsum('nchw,mc->nmhw', img1_trg, aud1).flatten(-2, -1) / self.tau, dim=1)
        a2i_2 = F.softmax(torch.einsum('nchw,mc->nmhw', img2_trg, aud2).flatten(-2, -1) / self.tau, dim=2)
        a2i = torch.log((a2i_1 * a2i_2).sum(2))    # nm

        B = img1.shape[0]
        labels = torch.arange(B).long().to(img1.device)
        loss = F.cross_entropy(a2i, labels) + F.cross_entropy(i2a, labels)

        # Compute avl maps
        with torch.no_grad():
            if self.use_mom_eval:
                Savl1 = torch.einsum('nchw,nc->nhw', img1_trg, aud1_trg) / self.tau
                Savl2 = torch.einsum('nchw,nc->nhw', img2_trg, aud2_trg) / self.tau
            else:
                Savl1 = torch.einsum('nchw,nc->nhw', img1, aud1) / self.tau
                Savl2 = torch.einsum('nchw,nc->nhw', img2, aud2) / self.tau
            Savl = (Savl1 + Savl2) / 2
            
        return loss, Savl

        
class FNAC(nn.Module):
    def __init__(self, tau, dim,  dropout_img, dropout_aud):
        super(FNAC, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.img_dropout = nn.Dropout(p= dropout_img)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)
        self.aud_dropout = nn.Dropout(p= dropout_aud)

        self.high_conf_thresh = 0.6
        # self.low_conf_thresh = 0.4

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def calculate_loss(self, img, aud, name=None):
        aud_attn = (aud@aud.transpose(0,1)) / self.tau

        img_avg =  self.avgpool(img)[:,:,0,0]
        img_attn = (img_avg@img_avg.transpose(0,1)) / self.tau
        
        B = img.shape[0]
        h,w = img.shape[2], img.shape[3]
        
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau

        loc_map = Slogits[torch.arange(B), torch.arange(B)]
        loc_map = (loc_map - torch.amin(loc_map, (1,2), keepdim=True))/ \
        (torch.amax(loc_map, (1,2), keepdim=True) - torch.amin(loc_map, (1,2), keepdim=True) + 1e-5)

        # frg_feature = img * loc_map.unsqueeze(1)
        frg_feature = img * (loc_map>self.high_conf_thresh).unsqueeze(1) # foreground visual features
        frg_feature = frg_feature.flatten(-2, -1).mean(dim=-1) 
        frg_attn = (frg_feature@frg_feature.transpose(0,1)) / self.tau

        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)

        fnac_loss1 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(logits, dim=1)) # FNS-1
        fnac_loss2 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(frg_attn, dim=1)) # TNS
        fnac_loss3 = F.l1_loss(torch.softmax(img_attn, dim=1), torch.softmax(logits, dim=1)) # FNS-2
    
        return [loss, fnac_loss1, fnac_loss2,  fnac_loss3], Slogits

    def forward(self, image, audio, name=None):
        # Image b*3*h*w 
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_dropout(img)
        img = self.img_proj(img) # b*512*7*7
        img = nn.functional.normalize(img, dim=1)
        img_avg =  self.avgpool(img)[:,:,0,0]
#         img_avg = nn.functional.normalize(img_avg, dim=1)

        # Audio b*1*h*w
        aud = self.audnet(audio)
        aud = self.aud_dropout(aud)
        aud = self.aud_proj(aud) # b*512
        aud_prenorm = aud
        aud = nn.functional.normalize(aud, dim=1)

        # Compute loss
        loss, logits = self.calculate_loss(img, aud, name=name)

        # Compute avl maps
        with torch.no_grad():
            B = img.shape[0]
            Savl = logits[torch.arange(B), torch.arange(B)]

        return loss, Savl