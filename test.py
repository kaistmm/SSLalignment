import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model import EZVSL, SLAVC, FNAC
from model_lvs import AVENet
from model_ssltie import AVENet_ssltie
from datasets import get_test_dataset, inverse_normalize, get_test_dataset_fnac
from datasets_lvs import GetAudioVideoDataset
from datasets_ssltie import GetAudioVideoDataset_ssltie
import cv2
from tqdm import tqdm
import utils_lvs
from opts_ssltie import SSLTIE_args
import matplotlib.pyplot as plt
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--pth_name', type=str, default='', help='pth name')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='/mnt/lynx1/datasets/FlickrSoundNet/Flickr_annotations/', type=str)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.pth_name)
    viz_dir = os.path.join(args.model_dir, args.pth_name.split('.')[0], 'viz')
    os.makedirs(os.path.join(args.model_dir, args.pth_name.split('.')[0]),exist_ok=True)
    os.makedirs(viz_dir,exist_ok=True)

    # Models
    if 'ssltie' in args.pth_name:
        ssltie_args=SSLTIE_args()
        ssltie_args=ssltie_args.ssltie_args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = AVENet_ssltie(ssltie_args)
        audio_visual_model = audio_visual_model.cuda()
        audio_visual_model = nn.DataParallel(audio_visual_model)
        ssltie_args.test_gt_path = args.test_gt_path
        ssltie_args.testset = args.testset
        ssltie_args.dataset_mode = args.testset
        ssltie_args.visualize = args.visualize
        ssltie_args.pth_name = args.pth_name
        
    elif 'lvs' in args.pth_name or 'ours' in args.pth_name:
        import easydict
        lvs_args = easydict.EasyDict({
        "data_path" : '',
        "image_size": 224,
        "batch_size" : 64,
        "n_threads" : 10,
        "epsilon" : 0.65,
        "epsilon2" : 0.4,
        'tri_map' : True,
        'Neg' : True,
        'random_threshold' : 1,
        'soft_ep' : 1,
        'test_gt_path' : args.test_gt_path,
        'testset' : args.testset,
        'visualize' : args.visualize,
        'pth_name' : args.pth_name,
        })
        audio_visual_model= AVENet(lvs_args) 
    elif 'ezvsl' in args.pth_name or 'margin' in args.pth_name:
        audio_visual_model = EZVSL(0.03,512)
    elif 'slavc' in args.pth_name:
        audio_visual_model = SLAVC(0.03, 512, 0, 0, 0.9, 0.9, False, None)
    elif 'fnac' in args.pth_name:
        audio_visual_model = FNAC(0.03,512,0,0)
    else:
        audio_visual_model = EZVSL(0.03,512)

    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        audio_visual_model.cuda(args.gpu)
        object_saliency_model.cuda(args.gpu)
    if args.multiprocessing_distributed:
        audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
        object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

    # Load weights
    
    if 'ssltie' in args.pth_name:
        ckp_fn = os.path.join(model_dir)
        checkpoint = torch.load(ckp_fn, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        
        model_without_dp = audio_visual_model.module
        model_without_dp.load_state_dict(state_dict)
        
    elif 'lvs' in args.pth_name:
        ckp_fn = os.path.join(model_dir)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = nn.DataParallel(audio_visual_model)
        audio_visual_model = audio_visual_model.cuda()
        checkpoint = torch.load(ckp_fn)
        model_dict = audio_visual_model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']            
        model_dict.update(pretrained_dict)
        audio_visual_model.load_state_dict(model_dict)
        audio_visual_model.to(device)
    
    elif 'ours' in args.pth_name:
        ckp_fn = os.path.join(model_dir)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = nn.DataParallel(audio_visual_model)
        audio_visual_model = audio_visual_model.cuda()
        checkpoint = torch.load(ckp_fn)
        model_dict = audio_visual_model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {'module.'+x:pretrained_dict[x] for x in pretrained_dict}
        model_dict.update(pretrained_dict)
        audio_visual_model.load_state_dict(model_dict,strict=False)
        audio_visual_model.to(device)
    
    elif 'htf' in args.pth_name:
        print('HTF evaluation')
        ckp_fn = 'htf no load'
    else:
        ckp_fn = os.path.join(model_dir)
        if os.path.exists(ckp_fn):
            ckp = torch.load(ckp_fn, map_location='cpu')
            audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        else:
            print(f"Checkpoint not found: {ckp_fn}")
    print(ckp_fn,' loaded')
    # Dataloader
    if 'ssltie' in args.pth_name:
        testdataset = GetAudioVideoDataset_ssltie(ssltie_args, mode='test')
        ssltie_args.image_path = testdataset.image_path
    if 'margin' in args.pth_name or 'ezvsl' in args.pth_name or 'slavc' in args.pth_name:
        testdataset = get_test_dataset(args)
        args.image_path = testdataset.image_path
    if 'fnac' in args.pth_name:
        testdataset = get_test_dataset_fnac(args)
        args.image_path = testdataset.image_path
    if 'lvs' in args.pth_name or 'ours' in args.pth_name:
        testdataset = GetAudioVideoDataset(lvs_args,  mode='test')
        lvs_args.image_path = testdataset.image_path
    if 'htf' in args.pth_name:
        testdataset = get_test_dataset(args)
        args.image_path = testdataset.image_path
    
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    
    print("Loaded dataloader.")
    
    if args.visualize:
        os.makedirs(os.path.join('../unified_qualitatives'),exist_ok=True)
        os.makedirs(os.path.join('../unified_qualitatives',args.testset),exist_ok=True)
        os.makedirs(os.path.join('../unified_qualitatives',args.testset,args.pth_name),exist_ok=True)
    
    if 'lvs' in args.pth_name or 'ours' in args.pth_name:
        validate_lvs(testdataloader, audio_visual_model, object_saliency_model, viz_dir, lvs_args)
    elif 'ssltie' in args.pth_name:
        validate_ssltie(testdataloader, audio_visual_model, object_saliency_model, viz_dir, ssltie_args)
    elif 'htf' in args.pth_name:
        validate_htf(testdataloader, object_saliency_model, viz_dir, args)
    else:
        validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args)
    
@torch.no_grad()
def validate_htf(testdataloader, object_saliency_model, viz_dir, args):
    from sklearn.metrics import auc
    npydir = '/mnt/lynx2/users/arda/htf/localization_results/'
    object_saliency_model.train(False)
    
    evaluator = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    evaluator_adap = utils.Evaluator_iiou()
    
    iou = []
    iou_adap = []
    for step, (image, spec, bboxes, name) in enumerate(tqdm(testdataloader)):
#         print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
#         heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
#         heatmap_arr =  heatmap.data.cpu().numpy()
        
        heatmap_obj = object_saliency_model(image)
        heatmap_obj = heatmap_obj.data.cpu().numpy()
        for i in range(spec.shape[0]):
#             heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now = np.load(npydir+'/'+name[i]+'.npy')
            heatmap_now = utils_lvs.normalize_img(-heatmap_now)
            gt_map = bboxes['gt_map'][i].data.cpu().numpy()#testset_gt(args,name[i])
            pred = 1 - heatmap_now
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            
#             pred_av = np.zeros(pred.shape)
#             pred_av[pred>threshold] = 1
#             pred_av[pred<=threshold] = 0
#             ciou,inter,union = evaluator.cal_CIOU(pred_av,gt_map,0.5)
            evaluator.cal_CIOU(pred,gt_map,threshold)
#             iou.append(ciou)
            
            gt_nums = (gt_map!=0).sum()
            if int(gt_nums) == 0:
                gt_nums = int(pred.shape[0] * pred.shape[1])//2
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
            
#             pred_adap = np.zeros(pred.shape)
#             pred_adap[pred>threshold] = 1
#             pred_adap[pred<=threshold] = 0
#             ciou_adap,_,_ = evaluator_adap.cal_CIOU(pred_adap,gt_map,0.5)
            evaluator.cal_CIOU_adap(pred,gt_map,threshold)
#             iou_adap.append(ciou_adap)
            short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
            if short_name in evaluator.iou.keys():
                evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                evaluator.iou[short_name].append(evaluator.ciou[-1])
            else:
#                 import pdb;pdb.set_trace()
                evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                evaluator.iou[short_name] = [evaluator.ciou[-1]]
            
            heatmap_obj_now = cv2.resize(heatmap_obj[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_obj_now = utils_lvs.normalize_img(-heatmap_obj_now)
            
            pred_obj = 1 - heatmap_obj_now
            pred_av_obj = utils.normalize_img(pred * 0.4 + pred_obj * (1 - 0.4))
            threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
            evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold)
            threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
            evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold)
            
            
            if args.visualize:
                if args.testset == 'vggss':
                    frame_ori_name = name[i].replace('.mp4','')+'/image_050.jpg'
                    target_name = name[i].replace('.mp4','')+'.jpg'
                elif args.testset == 'flickr':
                    jpgname = [x for x in os.listdir(os.path.join(args.image_path,name[i]+'.mp4')) if '.jpg' in x][0]
                    frame_ori_name = name[i]+'.mp4/'+jpgname
                    target_name = name[i].replace('.mp4','')+'.jpg'
                elif args.testset =='s4':
                    frame_ori_name = name[i]
                    target_name = name[i].split('/')[-1].replace('.png','.jpg')
                else:
                    frame_ori_name = name[i]
                    target_name = name[i]
                frame_ori = Image.open(os.path.join(args.image_path,frame_ori_name))
                frame_ori = frame_ori.resize((224,224))
                plt.imshow(utils.overlay(frame_ori,heatmap[i][0].cpu().detach().numpy()).resize((224,224)))
                plt.xticks([])  # Remove x ticks
                plt.yticks([])  # Remove y ticks
                plt.axis('off')
                plt.savefig(os.path.join('../unified_qualitatives',args.testset,args.pth_name,target_name), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
            
    ciou = evaluator.finalize_cIoU()        
    auc,auc_adap = evaluator.finalize_AUC()
    iauc = evaluator.finalize_IAUC()       
    iauc_adap = evaluator.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    ciou = ciou * 100
    ciou_adap = ciou_adap * 100
    iiou = iiou * 100
    iiou_adap = iiou_adap * 100
    auc = auc * 100
    iauc = iauc * 100
    auc_adap = auc_adap * 100
    iauc_adap = iauc_adap * 100
    
    ciou_ogl = evaluator_ogl.finalize_cIoU()        
    auc_ogl,auc_adap_ogl = evaluator_ogl.finalize_AUC()
    ciou_ogl,ciou_adap_ogl,_,_ = ciou_ogl
    ciou_ogl = ciou_ogl*100
    ciou_adap_ogl = ciou_adap_ogl*100
    auc_ogl = auc_ogl*100
    
    auc_adap_ogl = auc_adap_ogl*100
    
    print('mIoU(persample) : {:.3f}'.format(np.mean(evaluator.ciou)))
    print('CIoU : {:.1f}'.format(ciou),'CIoU_adap : {:.1f}'.format(ciou_adap),'iIoU : {:.1f}'.format(iiou),'iIoU_adap : {:.1f}'.format(iiou_adap))
    print('AUC : {:.1f}'.format(auc),'AUC_adap : {:.1f}'.format(auc_adap),'iAUC : {:.1f}'.format(iauc),'iAUC_adap : {:.1f}'.format(iauc_adap))
    
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(iiou,iiou_adap,iauc,iauc_adap))

    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(iiou,iiou_adap,iauc,iauc_adap))
    
    print('OGL results')
    print('CIoU : {:.2f}'.format(ciou_ogl),'CIoU_adap : {:.2f}'.format(ciou_adap_ogl))
    print('AUC : {:.2f}'.format(auc_ogl),'AUC_adap : {:.2f}'.format(auc_adap_ogl))    

@torch.no_grad()
def validate_lvs(testdataloader, model, object_saliency_model, viz_dir, args):
    from sklearn.metrics import auc
    model.train(False)
    object_saliency_model.train(False)
    
#     evaluator = utils_lvs.Evaluator()
    evaluator = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    evaluator_adap = utils.Evaluator_iiou()
#     import pdb;pdb.set_trace()    

    evaluator_full = utils.EvaluatorFull()
    
    iou = []
    iou_adap = []
    for step, (image, spec, bboxes, name) in enumerate(tqdm(testdataloader)):
#         print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
        
        heatmap_obj_ = object_saliency_model(image)
        
        heatmap_forextend = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=True)
        heatmap_forextend = heatmap_forextend.data.cpu().numpy()
        
        if args.testset == 'vggss':
            'from here'
            heatmap_arr =  heatmap.data.cpu().numpy()
            heatmap_obj = heatmap_obj_.data.cpu().numpy()

    #         heatmap = nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=True)
            for i in range(spec.shape[0]):
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    #             heatmap_now = heatmap[i:i+1].cpu().numpy()
#                 heatmap_forextend = heatmap_now
                heatmap_now = utils_lvs.normalize_img(-heatmap_now)
                gt_map = bboxes['gt_map'][i].data.cpu().numpy()#testset_gt(args,name[i])
                pred = 1 - heatmap_now
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                evaluator.cal_CIOU(pred,gt_map,threshold)

                gt_nums = (gt_map!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(pred.shape[0] * pred.shape[1])//2
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
                evaluator.cal_CIOU_adap(pred,gt_map,threshold)

                short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                if short_name in evaluator.iou.keys():
                    evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                    evaluator.iou[short_name].append(evaluator.ciou[-1])
                else:
    #                 import pdb;pdb.set_trace()
                    evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                    evaluator.iou[short_name] = [evaluator.ciou[-1]]
            
                heatmap_obj_now = cv2.resize(heatmap_obj[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_obj_now = utils_lvs.normalize_img(-heatmap_obj_now)

                pred_obj = 1 - heatmap_obj_now
                pred_av_obj = utils.normalize_img(pred * 0.4 + pred_obj * (1 - 0.4))
                threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
                evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold)
                threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold)
                
#                 min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)
                
#                 n=heatmap_forextend[i,0].size
#                 bb = bboxes['bboxes'][i]
#                 bb = bb[bb[:, 0] >= 0].numpy().tolist()
#                 av_min, av_max = -1. / 0.03, 1. / 0.03
#                 scores_av = min_max_norm(heatmap_forextend[i,0], av_min, av_max)
#                 conf_av = np.sort(scores_av.flatten())[-n//4:].mean()
#                 pred_av = utils_lvs.normalize_img(scores_av)
#                 thr_av = np.sort(pred_av.flatten())[int(n*0.5)]
                
#                 evaluator_full.update(bb, gt_map, conf_av, pred_av, thr_av, name[i])


                
                if args.visualize:
                    if args.testset == 'vggss':
                        frame_ori_name = name[i].replace('.mp4','')+'/image_050.jpg'
                        target_name = name[i].replace('.mp4','')+'.jpg'
                    else:
                        frame_ori_name = name[i]
                        target_name = name[i]
                    frame_ori = Image.open(os.path.join(args.image_path,frame_ori_name))
                    frame_ori = frame_ori.resize((224,224))
                    plt.imshow(utils.overlay(frame_ori,heatmap[i][0].cpu().detach().numpy()).resize((224,224)))
                    plt.xticks([])  # Remove x ticks
                    plt.yticks([])  # Remove y ticks
                    plt.axis('off')
                    plt.savefig(os.path.join('../unified_qualitatives',args.testset,args.pth_name,target_name), bbox_inches='tight', pad_inches=0, transparent=True)
                    plt.close()
            'to here'  
        
        else:
            heatmap = nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=True)
            heatmap_obj = nn.functional.interpolate(heatmap_obj_, size=(224, 224), mode='bilinear', align_corners=True)
            
            for i in range(spec.shape[0]):
    #             heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = heatmap[i:i+1].cpu().numpy()
                heatmap_now = utils_lvs.normalize_img(-heatmap_now[0][0])
                gt_map = bboxes['gt_map'][i].data.cpu().numpy()#testset_gt(args,name[i])
                pred = 1 - heatmap_now
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                evaluator.cal_CIOU(pred,gt_map,threshold)

                gt_nums = (gt_map!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(pred.shape[0] * pred.shape[1])//2
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
                evaluator.cal_CIOU_adap(pred,gt_map,threshold)

                short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                if short_name in evaluator.iou.keys():
                    evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                    evaluator.iou[short_name].append(evaluator.ciou[-1])
                else:
    #                 import pdb;pdb.set_trace()
                    evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                    evaluator.iou[short_name] = [evaluator.ciou[-1]]
            
                heatmap_obj_now = heatmap_obj[i:i+1].cpu().numpy()
                heatmap_obj_now = utils_lvs.normalize_img(-heatmap_obj_now[0][0])
                pred_obj = 1 - heatmap_obj_now
                pred_av_obj = utils.normalize_img(pred * 0.4 + pred_obj * (1 - 0.4))
                
                threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
                evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold)
                threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold)

                
                if args.visualize:
                    if args.testset == 'vggss':
                        frame_ori_name = name[i].replace('.mp4','')+'/image_050.jpg'
                        target_name = name[i].replace('.mp4','')+'.jpg'
                    elif args.testset == 'flickr':
                        jpgname = [x for x in os.listdir(os.path.join(args.image_path,name[i])) if '.jpg' in x][0]
                        frame_ori_name = name[i]+'/'+jpgname
                        target_name = name[i].replace('.mp4','')+'.jpg'
                    elif args.testset == 's4':
                        frame_ori_name = name[i]
                        target_name = name[i].split('/')[-1].replace('.png','.jpg')
                    else:
                        frame_ori_name = name[i]
                        target_name = name[i]
                    frame_ori = Image.open(os.path.join(args.image_path,frame_ori_name))
                    frame_ori = frame_ori.resize((224,224))
                    plt.imshow(utils.overlay(frame_ori,heatmap[i][0].cpu().detach().numpy()).resize((224,224)))
                    plt.xticks([])  # Remove x ticks
                    plt.yticks([])  # Remove y ticks
                    plt.axis('off')
                    plt.savefig(os.path.join('../unified_qualitatives',args.testset,args.pth_name,target_name), bbox_inches='tight', pad_inches=0, transparent=True)
                    plt.close()
            
    ciou = evaluator.finalize_cIoU()        
    auc,auc_adap = evaluator.finalize_AUC()
    iauc = evaluator.finalize_IAUC()       
    iauc_adap = evaluator.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    ciou = ciou * 100
    ciou_adap = ciou_adap * 100
    iiou = iiou * 100
    iiou_adap = iiou_adap * 100
    auc = auc * 100
    iauc = iauc * 100
    auc_adap = auc_adap * 100
    iauc_adap = iauc_adap * 100
    
    ciou_ogl = evaluator_ogl.finalize_cIoU()        
    auc_ogl,auc_adap_ogl = evaluator_ogl.finalize_AUC()
    ciou_ogl,ciou_adap_ogl,_,_ = ciou_ogl
    ciou_ogl = ciou_ogl*100
    ciou_adap_ogl = ciou_adap_ogl*100
    auc_ogl = auc_ogl*100
    auc_adap_ogl = auc_adap_ogl*100
    
    print('mIoU(persample) : {:.3f}'.format(np.mean(evaluator.ciou)))
    print('CIoU : {:.1f}'.format(ciou),'CIoU_adap : {:.1f}'.format(ciou_adap),'iIoU : {:.1f}'.format(iiou),'iIoU_adap : {:.1f}'.format(iiou_adap))
    print('AUC : {:.1f}'.format(auc),'AUC_adap : {:.1f}'.format(auc_adap),'iAUC : {:.1f}'.format(iauc),'iAUC_adap : {:.1f}'.format(iauc_adap))
    
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(iiou,iiou_adap,iauc,iauc_adap))

    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(iiou,iiou_adap,iauc,iauc_adap))
    
    print('OGL results')
    print('CIoU : {:.2f}'.format(ciou_ogl),'CIoU_adap : {:.2f}'.format(ciou_adap_ogl))
    print('AUC : {:.2f}'.format(auc_ogl),'AUC_adap : {:.2f}'.format(auc_adap_ogl))
    

@torch.no_grad()
def validate_ssltie(testdataloader, model, object_saliency_model, viz_dir, args):
    from sklearn.metrics import auc
    model.train(False)
    object_saliency_model.train(False)
    
    evaluator = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    evaluator_adap = utils.Evaluator_iiou()
    
    iou = []
    iou_adap = []
    for step, (image, spec, bboxes, name) in enumerate(tqdm(testdataloader)):
#         print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
        heatmap_arr =  heatmap.data.cpu().numpy()
        
        heatmap_obj = object_saliency_model(image)
        heatmap_obj = heatmap_obj.data.cpu().numpy()
        for i in range(spec.shape[0]):
            heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now = utils_lvs.normalize_img(-heatmap_now)
            gt_map = bboxes['gt_map'][i].data.cpu().numpy()#testset_gt(args,name[i])
            pred = 1 - heatmap_now
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            
#             pred_av = np.zeros(pred.shape)
#             pred_av[pred>threshold] = 1
#             pred_av[pred<=threshold] = 0
#             ciou,inter,union = evaluator.cal_CIOU(pred_av,gt_map,0.5)
            evaluator.cal_CIOU(pred,gt_map,threshold)
#             iou.append(ciou)
            
            gt_nums = (gt_map!=0).sum()
            if int(gt_nums) == 0:
                gt_nums = int(pred.shape[0] * pred.shape[1])//2
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
            
#             pred_adap = np.zeros(pred.shape)
#             pred_adap[pred>threshold] = 1
#             pred_adap[pred<=threshold] = 0
#             ciou_adap,_,_ = evaluator_adap.cal_CIOU(pred_adap,gt_map,0.5)
            evaluator.cal_CIOU_adap(pred,gt_map,threshold)
#             iou_adap.append(ciou_adap)
            short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
            if short_name in evaluator.iou.keys():
                evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                evaluator.iou[short_name].append(evaluator.ciou[-1])
            else:
#                 import pdb;pdb.set_trace()
                evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                evaluator.iou[short_name] = [evaluator.ciou[-1]]
            
            heatmap_obj_now = cv2.resize(heatmap_obj[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_obj_now = utils_lvs.normalize_img(-heatmap_obj_now)
            
            pred_obj = 1 - heatmap_obj_now
            pred_av_obj = utils.normalize_img(pred * 0.4 + pred_obj * (1 - 0.4))
            threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
            evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold)
            threshold = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
            evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold)
            
            
            if args.visualize:
                if args.testset == 'vggss':
                    frame_ori_name = name[i].replace('.mp4','')+'/image_050.jpg'
                    target_name = name[i].replace('.mp4','')+'.jpg'
                elif args.testset == 'flickr':
                    jpgname = [x for x in os.listdir(os.path.join(args.image_path,name[i]+'.mp4')) if '.jpg' in x][0]
                    frame_ori_name = name[i]+'.mp4/'+jpgname
                    target_name = name[i].replace('.mp4','')+'.jpg'
                elif args.testset =='s4':
                    frame_ori_name = name[i]
                    target_name = name[i].split('/')[-1].replace('.png','.jpg')
                else:
                    frame_ori_name = name[i]
                    target_name = name[i]
                frame_ori = Image.open(os.path.join(args.image_path,frame_ori_name))
                frame_ori = frame_ori.resize((224,224))
                plt.imshow(utils.overlay(frame_ori,heatmap[i][0].cpu().detach().numpy()).resize((224,224)))
                plt.xticks([])  # Remove x ticks
                plt.yticks([])  # Remove y ticks
                plt.axis('off')
                plt.savefig(os.path.join('../unified_qualitatives',args.testset,args.pth_name,target_name), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
        
        
    ciou = evaluator.finalize_cIoU()        
    auc,auc_adap = evaluator.finalize_AUC()
    iauc = evaluator.finalize_IAUC()       
    iauc_adap = evaluator.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    ciou = ciou * 100
    ciou_adap = ciou_adap * 100
    iiou = iiou * 100
    iiou_adap = iiou_adap * 100
    auc = auc * 100
    iauc = iauc * 100
    auc_adap = auc_adap * 100
    iauc_adap = iauc_adap * 100
    
    ciou_ogl = evaluator_ogl.finalize_cIoU()        
    auc_ogl,auc_adap_ogl = evaluator_ogl.finalize_AUC()
    ciou_ogl,ciou_adap_ogl,_,_ = ciou_ogl
    ciou_ogl = ciou_ogl*100
    ciou_adap_ogl = ciou_adap_ogl*100
    auc_ogl = auc_ogl*100
    
    auc_adap_ogl = auc_adap_ogl*100
    
    print('mIoU(persample) : {:.3f}'.format(np.mean(evaluator.ciou)))
    print('CIoU : {:.1f}'.format(ciou),'CIoU_adap : {:.1f}'.format(ciou_adap),'iIoU : {:.1f}'.format(iiou),'iIoU_adap : {:.1f}'.format(iiou_adap))
    print('AUC : {:.1f}'.format(auc),'AUC_adap : {:.1f}'.format(auc_adap),'iAUC : {:.1f}'.format(iauc),'iAUC_adap : {:.1f}'.format(iauc_adap))
    
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(iiou,iiou_adap,iauc,iauc_adap))

    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(iiou,iiou_adap,iauc,iauc_adap))
    
    print('OGL results')
    print('CIoU : {:.2f}'.format(ciou_ogl),'CIoU_adap : {:.2f}'.format(ciou_adap_ogl))
    print('AUC : {:.2f}'.format(auc_ogl),'AUC_adap : {:.2f}'.format(auc_adap_ogl))

@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    evaluator_adap = utils.Evaluator()
    
    
    evaluator_full = utils.EvaluatorFull()
    
    
    for step, (image, spec, bboxes, name) in enumerate(tqdm(testdataloader)):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        # Compute S_AVL
        heatmap_av = audio_visual_model(image.float(), spec.float())[1].unsqueeze(1)
        if False:#'fnac' in args.pth_name or 'margin' in args.pth_name:
            heatmap_av_ = F.interpolate(heatmap_av, size=(224, 224), mode='bicubic', align_corners=False)
        else:
            heatmap_av_ = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av_.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        if False:#'fnac' in args.pth_name or 'margin' in args.pth_name:
            heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bicubic', align_corners=False)
        else:
            heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_av_obj = utils.normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha))

            gt_map = bboxes['gt_map'][i].data.cpu().numpy()

            thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
            evaluator_av.cal_CIOU(pred_av, gt_map, thr_av)
        
            gt_nums = (gt_map!=0).sum()
            if int(gt_nums) == 0:
                gt_nums = int(pred_av.shape[0] * pred_av.shape[1])//2
            thr_adap = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1])-int(gt_nums)] # adap
            evaluator_av.cal_CIOU_adap(pred_av, gt_map, thr_adap)

            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
            evaluator_ogl.cal_CIOU(pred_av_obj, gt_map, thr_av_obj)
            
            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
            evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,thr_av_obj)
            
#             min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)
                
#             n=heatmap_av[i,0].size
#             bb = bboxes['bboxes'][i]
#             bb = bb[bb[:, 0] >= 0].numpy().tolist()
#             av_min, av_max = -1. / 0.03, 1. / 0.03
#             scores_av = min_max_norm(heatmap_av[i,0], av_min, av_max)
#             conf_av = np.sort(scores_av.flatten())[-n//4:].mean()
#             pred_av__ = utils_lvs.normalize_img(scores_av)
#             thr_av__ = np.sort(pred_av.flatten())[int(n*0.5)]

#             evaluator_full.update(bb, gt_map, conf_av, pred_av__, thr_av__, name[i])

            
            
            

            short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
            if short_name in evaluator_av.iou.keys():
                evaluator_av.iou_adap[short_name].append(evaluator_av.ciou_adap[-1])
                evaluator_av.iou[short_name].append(evaluator_av.ciou[-1])
            else:
#                 import pdb;pdb.set_trace()
                evaluator_av.iou_adap[short_name] = [evaluator_av.ciou_adap[-1]]
                evaluator_av.iou[short_name] = [evaluator_av.ciou[-1]]
        
        
            if args.visualize:
#                 import pdb;pdb.set_trace()
                if args.testset == 'vggss':
                    frame_ori_name = name[i].replace('.mp4','')+'/image_050.jpg'
                    target_name = name[i].replace('.mp4','')+'.jpg'
                elif args.testset =='s4':
                    frame_ori_name = name[i]+'.png'
#                     import pdb;pdb.set_trace()
                    target_name = name[i].split('/')[-1].replace('.png','')+'.jpg'
                else:
                    frame_ori_name = name[i]+'.jpg'
                    target_name = name[i]+'.jpg'
                frame_ori = Image.open(os.path.join(args.image_path,frame_ori_name))
                frame_ori = frame_ori.resize((224,224))
                plt.imshow(utils.overlay(frame_ori,heatmap_av_[i][0].cpu().detach().numpy()).resize((224,224)))
                plt.xticks([])  # Remove x ticks
                plt.yticks([])  # Remove y ticks
                plt.axis('off')
                plt.savefig(os.path.join('../unified_qualitatives',args.testset,args.pth_name,target_name), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
            
    ciou = evaluator_av.finalize_cIoU()        
    auc,auc_adap = evaluator_av.finalize_AUC()
    iauc = evaluator_av.finalize_IAUC()       
    iauc_adap = evaluator_av.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    ciou = ciou * 100
    ciou_adap = ciou_adap * 100
    iiou = iiou * 100
    iiou_adap = iiou_adap * 100
    auc = auc * 100
    iauc = iauc * 100
    auc_adap = auc_adap * 100
    iauc_adap = iauc_adap * 100
    
    ciou_ogl = evaluator_ogl.finalize_cIoU()        
    auc_ogl,auc_adap_ogl = evaluator_ogl.finalize_AUC()
    ciou_ogl,ciou_adap_ogl,_,_ = ciou_ogl
    ciou_ogl = ciou_ogl*100
    ciou_adap_ogl = ciou_adap_ogl*100
    auc_ogl = auc_ogl*100
    auc_adap_ogl = auc_adap_ogl*100
    
    print('mIoU(persample) : {:.3f}'.format(np.mean(evaluator_av.ciou)))
    print('CIoU : {:.1f}'.format(ciou),'CIoU_adap : {:.1f}'.format(ciou_adap),'iIoU : {:.1f}'.format(iiou),'iIoU_adap : {:.1f}'.format(iiou_adap))
    print('AUC : {:.1f}'.format(auc),'AUC_adap : {:.1f}'.format(auc_adap),'iAUC : {:.1f}'.format(iauc),'iAUC_adap : {:.1f}'.format(iauc_adap))
    
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(iiou,iiou_adap,iauc,iauc_adap))

    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(ciou,ciou_adap,auc,auc_adap))
    print('{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(iiou,iiou_adap,iauc,iauc_adap))
    
    print('OGL results')
    print('CIoU : {:.2f}'.format(ciou_ogl),'CIoU_adap : {:.2f}'.format(ciou_adap_ogl))
    print('AUC : {:.2f}'.format(auc_ogl),'AUC_adap : {:.2f}'.format(auc_adap_ogl))
    
    print(ciou,auc,auc_adap,iauc,iauc_adap)

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


if __name__ == "__main__":
    main(get_arguments())
