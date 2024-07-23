import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av


def load_image(path):
    return Image.open(path).convert('RGB')

def load_spectrogram_fnac(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)
    
    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_spectrogram(path, dur=10.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    # audio_dur = min(float(audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base), dur)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=0, duration=dur)  # Get full audio

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    audio_center = int(float(audio_dur) / 2 * samplerate)
    shift_index = int(samplerate * dur / 2) - audio_center
    audio = np.roll(audio, shift_index)

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    # frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=354)
    # spectrogram = spectrogram[:, :-1]
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram


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

class AudioVisualDataset(Dataset):
    def __init__(self, args, image_files, audio_files, image_path, audio_path, audio_dur=3., image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr', model_name='ezvsl', img_selection=None, dual=False):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = np.array(audio_files)
        self.image_files = np.array(image_files)
        self.all_bboxes = all_bboxes
        
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform

        self.img_selection = img_selection  # For flickr image selection
        self.model_name = model_name
        self.args = args
        
    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]
        if self.args.testset == 'vggss':
            file_id = file.split('/')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        if self.img_selection == 'random':
            item = random.choice(os.listdir(os.path.join(self.image_path, file_id)))
            img_fn = os.path.join(self.image_path, file_id, item)
        if self.img_selection == 'center':
            items = os.listdir(os.path.join(self.image_path, file_id))
            item = items[int(len(items)/2)]
            img_fn = os.path.join(self.image_path, file_id, item)
        if self.img_selection == 'first':
            items = os.listdir(os.path.join(self.image_path, file_id))
            item = items[0]
            img_fn = os.path.join(self.image_path, file_id, item)
        
#         if self.model_name == 'fnac' and self.bbox_format=='flickr':
#             mp4_path = img_fn + '.mp4'
#             jpg = os.listdir(mp4_path)
#             jpg = [x for x in jpg if x[-3:]=='jpg'][0]
#             img_fn = os.path.join(mp4_path,jpg)
        

        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])+'.wav'
        
        if self.model_name == 'fnac':
            spectrogram = self.audio_transform(load_spectrogram_fnac(audio_fn))
        else:
            spectrogram = self.audio_transform(load_spectrogram(audio_fn, dur = self.audio_dur))

        bboxes = {}
        if self.all_bboxes is not None:
#             bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        return frame, spectrogram, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # try:
        return self.getitem(idx)
        # except Exception:
        #     return self.getitem(random.sample(range(len(self)), 1)[0])

def get_test_dataset(args):
    if args.testset == 'flickr':
        audio_path = '/mnt/lynx2/users/gonhy/audio_250/'#'/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_wav_test/'
        image_path = '/mnt/lynx2/users/gonhy/frames_250/'#'/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_img_test/'
    elif args.testset == 'vggss':
        audio_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_aud/'
        image_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_img/'
    elif args.testset == 'is3':
        audio_path = '/mnt/lynx2/users/arda/synthetic3241/audio_wav'
        image_path = '/mnt/lynx2/users/arda/synthetic3241/visual_frames'
    
    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/ours_vggss.txt'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    elif args.testset == 'is3':
        testcsv = 'metadata/synthetic3240_bbox.json'
    elif args.testset == 'vposs':
        testcsv = 'metadata/vpo_ss_bbox.json'
    elif args.testset == 'vpoms':
        testcsv = 'metadata/vpo_ms_bbox.json'
    elif args.testset == 'ms3':
        testcsv = 'metadata/ms3_box.json'
    elif args.testset == 's4':
        testcsv = 'metadata/s4_box.json'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss',
                   'is3':'is3',
                   'ms3':'ms3',
                   's4':'s4',
                   'vpoms':'vpoms',
                   'vposs':'vposs'
                  }[args.testset]
    audio_length = 10.0

    #  Retrieve list of audio and video files
    
    if 'json' in testcsv:
        with open(testcsv) as fi:
            jsonfile = json.load(fi)
        
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
        
        if args.testset == 'ms3':
            audio_length = 5.0
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]
            
        elif args.testset == 's4':
            audio_length = 5.0
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])

            audio_files = ['/'.join(fn['audio'].split('/')[-2:])[:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]
            
        else:    
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]
    else:
        testset = set([item[0] for item in csv.reader(open(testcsv))])

        # Intersect with available files
        audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
        avail_files = audio_files.intersection(image_files)
        testset = testset.intersection(avail_files)

        testset = sorted(list(testset))
        
        if args.testset == 'flickr':
            image_files = [dt+'.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
        elif args.testset == 'vggss':
            image_files = [dt+'/image_050.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
            

        # Bounding boxes
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # Transforms
#     image_transform = transforms.Compose([
#         transforms.Resize((224, 224), Image.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    spec_size = (257, 200)
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.CenterCrop(spec_size),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        args=args,
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=audio_length,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format
    )

def get_test_dataset_fnac(args):
    if args.testset == 'flickr':
        audio_path = '/mnt/lynx2/users/gonhy/audio_250/'#'/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_wav_test/'
        image_path = '/mnt/lynx2/users/gonhy/frames_250/'#'/mnt/lynx1/datasets/FlickrSoundNet/Flickr_Sound_Top5_Dataset_img_test/'
    elif args.testset == 'vggss':
        audio_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_aud/'
        image_path = '/mnt/lynx1/datasets/VGGSound_v1/VGGSound_img/'
    elif args.testset == 'is3':
        audio_path = '/mnt/lynx2/users/arda/synthetic3241/audio_wav'
        image_path = '/mnt/lynx2/users/arda/synthetic3241/visual_frames'
    
    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/ours_vggss.txt'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    elif args.testset == 'is3':
        testcsv = 'metadata/synthetic3240_bbox.json'
    elif args.testset == 'vposs':
        testcsv = 'metadata/vpo_ss_bbox.json'
    elif args.testset == 'vpoms':
        testcsv = 'metadata/vpo_ms_bbox.json'
    elif args.testset == 'ms3':
        testcsv = 'metadata/ms3_box.json'
    elif args.testset == 's4':
        testcsv = 'metadata/s4_box.json'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss',
                   'is3':'is3',
                   'ms3':'ms3',
                   's4':'s4',
                   'vpoms':'vpoms',
                   'vposs':'vposs'
                  }[args.testset]

    #  Retrieve list of audio and video files
#     testset = set([item[0] for item in csv.reader(open(testcsv))])

#     # Intersect with available files
#     audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
#     image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
#     avail_files = audio_files.intersection(image_files)
#     testset = testset.intersection(avail_files)
    
    if 'json' in testcsv:
        with open(testcsv) as fi:
            jsonfile = json.load(fi)
        
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
        
        if args.testset == 'ms3':
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]
            
        elif args.testset == 's4':
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])

            audio_files = ['/'.join(fn['audio'].split('/')[-2:])[:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]
            
        else:    
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]
    
    else:
        testset = set([item[0] for item in csv.reader(open(testcsv))])

        # Intersect with available files
        audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
        avail_files = audio_files.intersection(image_files)
        testset = testset.intersection(avail_files)

        testset = sorted(list(testset))
        
        if args.testset == 'flickr':
            image_files = [dt+'.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
        elif args.testset == 'vggss':
            image_files = [dt+'/image_050.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
#         if args.testset == 'flickr':
#             image_files = []
#             audio_files = []
#             filelist = os.listdir(audio_path)
#             filelist = sorted(filelist)
#             for item in filelist:
#                 image_files.append(item.replace('.wav',''))
#                 audio_files.append(item.replace('.wav',''))

#         elif args.testset == 'vggss':
#             testset = set([item[0] for item in csv.reader(open(testcsv))])

#             # Intersect with available files
#             audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
#             image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
#             avail_files = audio_files.intersection(image_files)
#             testset = testset.intersection(avail_files)

#             testset = sorted(list(testset))

#             image_files = [dt+'/image_050.jpg' for dt in testset]
#             audio_files = [dt for dt in testset]    
            
        # Bounding boxes
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # get all classes
    # all_classes = {}
    # if args.testset == "vggss":
    #     all_classes = get_all_classes()

    # Transforms
#     image_transform = transforms.Compose([
#         transforms.Resize((224, 224), Image.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        args=args,
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
        model_name='fnac'
    )

def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



