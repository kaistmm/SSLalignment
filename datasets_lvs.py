import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf
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



class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        
        if args.testset == 'flickr':
            self.audio_path = '/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_wav_test/'
            self.image_path = '/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_img_test/'
        else:
            self.audio_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_aud/'
            self.image_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_img/'
        
        self.imgSize = args.image_size 
        self.args = args
        
        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []

        data = []
        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/ours_vggss.txt'
        elif args.testset == 'is3':
            testcsv = 'metadata/synthetic3240_bbox.json'
        elif args.testset == 'ms3':
            testcsv = 'metadata/ms3_box.json'
        elif args.testset == 's4':
            testcsv = 'metadata/s4_box.json'
        elif args.testset == 'vposs':
            testcsv = 'metadata/vpo_ss_bbox.json'
        elif args.testset == 'vpoms':
            testcsv = 'metadata/vpo_ms_bbox.json'
        
        self.audio_length = 10
        self.st = 3.5
        self.fi = 6.5
            
        if 'json' in testcsv:
            with open(testcsv) as fi:
                jsonfile = json.load(fi)

            self.all_bboxes = load_all_bboxes(args.test_gt_path, format=args.testset)
            
            if args.testset == 'ms3':
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]

                self.video_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])
            elif args.testset == 's4':
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = ['/'.join(fn['audio'].split('/')[-2:]) for fn in jsonfile]
                image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]

                self.video_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
                self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])
            else:
                self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                self.video_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])
            
            
        else:
            with open(testcsv) as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    data.append(item[0] + '.mp4')
            self.all_bboxes = load_all_bboxes(args.test_gt_path, format=args.testset)
            if args.testset == 'vggss':
                exists = os.listdir('/mnt/lynx1/datasets/VGGSound_v1/VGGSound_img/')
                exists = set(exists)-set(['7XQN9XDnRm4_80000_90000'])
                exists = set([x+'.mp4' for x in exists])
#                 exists = set([x[:12]+str(int(x[12:].split('_')[0])//1000).zfill(6)+'.mp4' for x in exists])
                data = set(data).intersection(exists)
            for item in data:
                self.video_files.append(item )

                
        print(len(self.video_files))
        self.count = 0
        

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])      ## where we got 85.2 on flickr
            '''Does order of normalization matters?'''
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
#  

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]

        # Image
        
        if self.args.testset == 'flickr':
            mp4_path = self.image_path + file[:-4] + '.mp4'
            jpg = os.listdir(mp4_path)
            jpg = [x for x in jpg if x[-3:]=='jpg'][0]
            filename = os.path.join(mp4_path,jpg)
            audiofilename = self.audio_path + file[:-3]+'wav'
        elif self.args.testset == 'vggss':
            filetmp = file.replace('.mp4','')
#             filetmp = filetmp[:11]+'_'+str(int(filetmp[12:])*1000)+'_'+str((int(filetmp[12:])+10)*1000)
            filename = os.path.join(self.image_path,filetmp,'image_050.jpg')
            audiofilename = os.path.join(self.audio_path,filetmp+'.wav')
        elif self.args.testset == 'is3' or self.args.testset == 'vposs' or self.args.testset == 'vpoms':
            filename = os.path.join(self.image_path,file)
            audiofilename = os.path.join(self.audio_path,self.audio_files[idx])
        elif self.args.testset == 'ms3' or self.args.testset == 's4':
            filename = os.path.join(self.image_path,file)
            audiofilename = os.path.join(self.audio_path,self.audio_files[idx])
            
        frame = self.img_transform(self._load_frame(filename))
        frame_ori = np.array(self._load_frame(filename))
        # Audio
        samples, samplerate = sf.read(audiofilename)
        
        if len(samples.shape) > 1:
            if samples.shape[1] == 2:
                samples = samples[:,0]

        # repeat if audio is too short
        if samples.shape[0] < samplerate * self.audio_length:
            n = int(samplerate * self.audio_length / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*self.audio_length]
        resamples = resamples[int(16000*self.st):int(16000*self.fi)]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
        
        bboxes = {}
        if self.all_bboxes is not None:
            bb = -torch.ones((10, 4)).long()
            tmpbox = self.all_bboxes[file[:-4]]
            bb[:len(tmpbox)] = torch.from_numpy(np.array(tmpbox))
            bboxes['bboxes'] = bb
            
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file[:-4]], self.args.testset)
        
        return frame,spectrogram,bboxes,file



