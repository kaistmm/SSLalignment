import os
import cv2
import json
from scipy.stats.stats import mode
import torch
import csv
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchaudio
import torchaudio.transforms as audio_T
import pdb
import time
from PIL import Image, ImageFilter
import glob
import sys
import warnings
warnings.filterwarnings("ignore")
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf
torchaudio.set_audio_backend("sox_io")
sys.path.append('./datasets/')
import pdb
import xml.etree.ElementTree as ET

def vgg_filename(name):
    return '_'.join([name[:11],str(int(name[12:])*1000),str((int(name[12:])+10)*1000)])
    

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            filename = vgg_filename(annotation['file'])
            gt_bboxes[filename] = bboxes

    if format == 'is3':
        with open('metadata/synthetic3240_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'vposs':
        with open('metadata/vpo_ss_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'vpoms':
        with open('metadata/vpo_ms_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'ms3':
        with open('metadata/ms3_box.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes['/'.join(annotation['image'].split('/')[-2:])[:-4]] = bboxes
    if format == 's4':
        with open('metadata/s4_box.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes['/'.join(annotation['image'].split('/')[-3:])[:-4]] = bboxes
    
    

    return gt_bboxes



def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    else:#if format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GetAudioVideoDataset_ssltie(Dataset):

    def __init__(self, args, mode='train', transforms=None):
 
        data = []
    
        self.imgSize = args.image_size 

        self.AmplitudeToDB = audio_T.AmplitudeToDB()

        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []
        self.args = args
        self.audio_length = 10

        if args.dataset_mode == 'vggss':
            args.trainset_path = args.trainset_path
        elif args.dataset_mode == 'flickr':
            args.trainset_path = args.Flickr_trainset_path
            
        self.all_bboxes = load_all_bboxes(args.test_gt_path, format=args.testset)

        if args.testset == 'is3':
            testcsv = 'metadata/synthetic3240_bbox.json'
        elif args.testset == 'vposs':
            testcsv = 'metadata/vpo_ss_bbox.json'
        elif args.testset == 'vpoms':
            testcsv = 'metadata/vpo_ms_bbox.json'
        elif args.testset == 'ms3':
            testcsv = 'metadata/ms3_box.json'
        elif args.testset == 's4':
            testcsv = 'metadata/s4_box.json'
        
        # Debug with a small dataset
        if args.debug:
            
            if mode=='train':
                with open('metadata/debug_data/train_vggss_debug_100.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.rstrip('\n'))
                    self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                    self.video_path = args.trainset_path + '/total_video_frames/'
            
            elif mode=='test':
                with open('metadata/debug_data/test_vggss_debug_50.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.split('.')[0])
                    self.audio_path = args.vggss_test_path + '/audio/'
                    self.video_path = args.vggss_test_path + '/frame/'

            elif mode=='val':
                with open('metadata/test_flick.csv') as f:
                    csv_reader = csv.reader(f)
                    for item in csv_reader:
                        data.append(item[0])
                    
                    self.audio_path = args.soundnet_test_path + '/audio/'
                    self.video_path = args.soundnet_test_path + '/frames/'

        else:
            if args.dataset_mode == 'vggss':
                if mode=='train':
                    if self.args.training_set_scale == 'subset_144k':
                        if self.args.ret_seen_144k:
                            train_list_file = 'train_seen_144k_list.txt'
                        else:
                            train_list_file = 'train_vggss_train_144k.txt'
                    elif self.args.training_set_scale == 'subset_10k':
                        train_list_file = 'train_vggss_train_10k.txt'
                    else:
                        train_list_file = 'train_vggss_190228.txt' 

                    with open('metadata/' + train_list_file,'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.rstrip('\n'))
                        self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                        self.video_path = args.trainset_path + '/total_video_frames/'

                elif mode=='test':
#                     with open('metadata/test_vggss_4911.txt','r') as f:
#                         txt_reader = f.readlines()
#                         for item in txt_reader[:]:
#                             data.append(item.split('.')[0])
#                         self.audio_path = args.vggss_test_path + '/audio/'
#                         self.video_path = args.vggss_test_path + '/frame/'
                    with open('metadata/ours_vggss.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader[:]:
                            data.append(item.split('\n')[0])
                        self.audio_path = args.vggss_test_path + 'audios_wav'
                        self.video_path = args.vggss_test_path + 'VGGSound_img'
                        
                elif mode=='test_train':
#                     with open('metadata/test_vggss_4911.txt','r') as f:
#                         txt_reader = f.readlines()
#                         for item in txt_reader[:]:
#                             data.append(item.split('.')[0])
#                         self.audio_path = args.vggss_test_path + '/audio/'
#                         self.video_path = args.vggss_test_path + '/frame/'
                    with open('/home/arda/iccv/tpami/ours_140k.txt','r') as f:
                        for s in f.readlines():
                            data.append(s.replace('\n',''))
                        self.audio_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_aud/'
                        self.video_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_img/'
                        
                elif mode=='val':
                    with open('metadata/test_flick.csv') as f:
                        # if arg.test == 'test.csv':
                        csv_reader = csv.reader(f)
                        for item in csv_reader:
                            data.append(item[0])
                        
                        self.audio_path = args.soundnet_test_path + '/audio/'
                        self.video_path = args.soundnet_test_path + '/frame/'
            
            elif args.dataset_mode == 'flickr':
                if mode == 'train':
                    if self.args.training_set_scale == 'subset_10k':
                        train_list_file = 'train_flickr_10k.txt'
                    else:
                        train_list_file = 'train_flickr_144k.txt'
                    with open('metadata/' + train_list_file,'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.rstrip('\n'))
                        self.audio_path = args.trainset_path + '/mp3/'
                        self.video_path = args.trainset_path + '/frames/'
                
                elif mode == 'test':
                    with open('metadata/test_flickr_250.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.split('.')[0])
                        self.audio_path = args.soundnet_test_path + '/Flickr_Sound_Top5_Dataset_wav_test'
                        self.video_path = args.soundnet_test_path + '/Flickr_Sound_Top5_Dataset_img_test'
            elif args.dataset_mode == 'is3' or args.dataset_mode == 'vposs' or args.dataset_mode == 'vpoms':
                print(testcsv)
                with open(testcsv) as fi:
                    jsonfile = json.load(fi)
                    
                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                self.video_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])

                self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                data = [fn['image'].split('/')[-1] for fn in jsonfile]
            elif args.testset == 'ms3':
                self.audio_length = 5
                with open(testcsv) as fi:
                    jsonfile = json.load(fi)
                    
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                data = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                self.video_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])
                
            elif args.testset == 's4':
                self.audio_length = 5
                with open(testcsv) as fi:
                    jsonfile = json.load(fi)
                    
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = ['/'.join(fn['audio'].split('/')[-2:]) for fn in jsonfile]
                data = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
                self.video_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])
            
        self.image_path = self.video_path
        for item in data[:]:
            self.video_files.append(item )

        print("{0} dataset size: {1}".format(self.mode.upper() , len(self.video_files)))
        
        self.count = 0
        
        
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if self.mode == 'train':

            if self.args.img_aug == 'moco_v1':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

                self.img_transform = transforms.Compose(augmentation)

            elif self.args.img_aug == 'moco_v2':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.3, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
                
                self.img_transform = transforms.Compose(augmentation)
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.imgSize,self.imgSize), Image.BICUBIC),
                transforms.CenterCrop((self.imgSize,self.imgSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now           

    def _init_atransform(self):
        # self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
        self.aid_transform = transforms.Compose([transforms.ToTensor()])


    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]

        if self.args.dataset_mode == 'vggss':
            if self.mode == 'train':
                frame = self.img_transform(self._load_frame(os.path.join( self.video_path, file , '125.jpg' ) ))
                frame_ori = np.array(self._load_frame(os.path.join( self.video_path, file , '125.jpg' ) ))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))

            elif self.mode in ['test', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '/image_050.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '/image_050.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '/'+file+'.wav'))
                
            elif self.mode in ['test_train', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '/image_050.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '/image_050.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file+'.wav'))
        
        ### For Flickr_SoundNet training: 
        elif self.args.dataset_mode == 'flickr':
            if self.mode == 'train':
                frame = self.img_transform(self._load_frame(os.path.join(self.video_path, file, '00000003.jpg')))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file, '00000003.jpg') ))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.mp3'))
                # Only the first four seconds of the audio is used, when training
                samples = samples[...,:samplerate * 4]

            elif self.mode in ['test', 'val'] :
                mp4_path = os.path.join(self.video_path , file + '.mp4')
                jpg = os.listdir(mp4_path)
                jpg = [x for x in jpg if x[-3:]=='jpg'][0]
                filename = os.path.join(mp4_path,jpg)
                
                frame = self.img_transform(self._load_frame(filename ))
                frame_ori = np.array(self._load_frame(filename))
                
#                 frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
#                 frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
        elif self.args.dataset_mode == 'is3' or self.args.dataset_mode == 'vpoms' or self.args.dataset_mode == 'vposs':
            frame = self.img_transform(self._load_frame(os.path.join(self.video_path,file)))
            samples, samplerate = torchaudio.load(os.path.join(self.audio_path,self.audio_files[idx]))
        elif self.args.testset == 'ms3' or self.args.testset == 's4':
            frame = self.img_transform(self._load_frame(os.path.join(self.video_path,file)))
            samples, samplerate = torchaudio.load(os.path.join(self.audio_path,self.audio_files[idx]))
            if samples.shape[0] == 2:
                samples = samples[0:1,:]
        
        if samples.shape[1] < samplerate * self.audio_length:
            n = int(samplerate * self.audio_length / samples.shape[1]) + 1
            samples = samples.repeat(1, n)

        samples = samples[...,:samplerate*self.audio_length]

        spectrogram  =  audio_T.MelSpectrogram(
                sample_rate=samplerate,
                n_fft=512,
                hop_length=239, 
                n_mels=257,
                normalized=True
            )(samples)
        
        if (self.args.aud_aug=='SpecAug') and (self.mode=='train') and (random.random() < 0.8):
            maskings = nn.Sequential(
                audio_T.TimeMasking(time_mask_param=180),
                audio_T.FrequencyMasking(freq_mask_param=35)
                )
            spectrogram = maskings(spectrogram)

        spectrogram = self.AmplitudeToDB(spectrogram)
        
        bboxes = {}
        if self.all_bboxes is not None:
#             bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file.split('.')[0]], self.args.testset)
            
        return frame, spectrogram, bboxes, file



