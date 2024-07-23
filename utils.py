import os
import json
from torch.optim import *
import numpy as np
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from PIL import Image

class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer > thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []
        
from sklearn import metrics

class Evaluator_iiou(object):
    def __init__(self):
        super(Evaluator_iiou, self).__init__()
        self.ciou = []
        self.ciou_adap = []
        self.iou = {}
        self.iou_adap = {}
        self.iiou = []
        self.iiou_adap = []
        
    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer > thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def cal_CIOU_adap(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer > thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou_adap.append(ciou)
        
    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        
        cious_adap = [np.sum(np.array(self.ciou_adap) >= 0.05*i) / len(self.ciou_adap)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc_adap = metrics.auc(thr, cious_adap)
        return auc,auc_adap
    
    def finalize_IAUC(self):
        iious = []
        for i in range(21):
            iiou = []
            for name in self.iou:
                iou_pair = np.array(self.iou[name])
                iou_pair[iou_pair>=0.05*i]=1
                iou_pair[iou_pair<0.05*i]=0
                iiou.append(np.min(iou_pair)) 
            iious.append(np.mean(iiou))
            
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, iious)
        return auc
    
    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_IAUC_adap(self):
        iious = []
        for i in range(21):
            iiou_adap = []
            for name in self.iou_adap:
                iou_pair = np.array(self.iou_adap[name])
                iou_pair[iou_pair>=0.05*i]=1
                iou_pair[iou_pair<0.05*i]=0
                iiou_adap.append(np.min(iou_pair)) 
            iious.append(np.mean(iiou_adap))
            
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, iious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou05 = np.sum(np.array(self.ciou) >= 0.5)/len(self.ciou)
        ciou_adap = np.sum(np.array(self.ciou_adap) >= 0.5)/len(self.ciou_adap)
        ciou = ciou05
        
        
        for name in self.iou:
            iou_pair = np.array(self.iou[name])
            iou_pair[iou_pair>=0.5]=1
            iou_pair[iou_pair<0.5]=0
#             if np.min(iou_pair) == 1:
#                 print(name)
            self.iiou.append(np.min(iou_pair))
        for name in self.iou_adap:
            iou_pair = np.array(self.iou_adap[name])
            iou_pair[iou_pair>=0.5]=1
            iou_pair[iou_pair<0.5]=0
            self.iiou_adap.append(np.min(iou_pair)) 
        
        
        return [ciou,ciou_adap,np.mean(self.iiou),np.mean(self.iiou_adap)]

    def clear(self):
        self.ciou = []

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def visualize(raw_image, boxes):
    import cv2
    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:

        xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_iou(iou_list, suffix, output_dir):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(output_dir,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()

def overlay(img, heatmap, cmap = 'jet', alpha=0.5):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)
        #Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes = True)
        heatmap = Image.fromarray(heatmap[:,:,:3],mode='RGB')
    # Resize the heatmap to cover whole img
    heatmap = heatmap.resize((img.size[0], img.size[1]), resample = Image.BILINEAR)
    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    return result



class EvaluatorFull(object):
    def __init__(self, iou_thrs=[0.5], default_conf_thr=0.5, pred_size=0.5, pred_thr=0.5):
        super(EvaluatorFull, self).__init__()
        self.iou_thrs = iou_thrs
        self.default_conf_thr = default_conf_thr
        self.min_sizes = {'small': 0, 'medium': 32**2, 'large': 96**2, 'huge': 144**2}
        self.max_sizes = {'small': 32**2, 'medium': 96**2, 'large': 144**2, 'huge': 10000**2}

        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []

    @staticmethod
    def calc_precision_recall(bb_list, ciou_list, confidence_list, confidence_thr, ciou_thr=0.5):
        assert len(bb_list) == len(ciou_list) == len(confidence_list)
        true_pos, false_pos, false_neg = 0, 0, 0
        for bb, ciou, confidence in zip(bb_list, ciou_list, confidence_list):
            if bb == 0:
                # no sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected
                    false_pos += 1
            else:
                # sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected...
                    if ciou >= ciou_thr:    # ...in correct place
                        true_pos += 1
                    else:                   # ...in wrong place
                        false_pos += 1
                else:
                    # no sounding objects detected
                    false_neg += 1 

        precision = 1. if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
        recall = 1. if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)

        return precision, recall

    def calc_ap(self, bb_list_full, ciou_list_full, confidence_list_full, iou_thr=0.5):

        assert len(bb_list_full) == len(ciou_list_full) == len(confidence_list_full)

        # for visible objects
        # ss = [i for i, bb in enumerate(bb_list_full) if bb > 0]
        # bb_list = [bb_list_full[i] for i in ss]
        # ciou_list = [ciou_list_full[i] for i in ss]
        # confidence_list = [confidence_list_full[i] for i in ss]

        precision, recall, skip_thr = [], [], max(1, len(ciou_list_full)//200)
        for thr in np.sort(np.array(confidence_list_full))[:-1][::-skip_thr]:
            p, r = self.calc_precision_recall(bb_list_full, ciou_list_full, confidence_list_full, thr, iou_thr)
            precision.append(p)
            recall.append(r)
        precision_max = [np.max(precision[i:]) for i in range(len(precision))]
        ap = sum([precision_max[i]*(recall[i+1]-recall[i])
                  for i in range(len(precision_max)-1)])
        return ap

    def cal_auc(self, bb_list, ciou_list):
        ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        ciou = [ciou_list[i] for i in ss]
        cious = [np.sum(np.array(ciou) >= 0.05*i) / len(ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def filter_subset(self, subset, name_list, area_list, bb_list, ciou_list, conf_list):
        import pdb;pdb.set_trace()
        if subset == 'visible':
            ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        elif subset == 'non-visible/non-audible':
            ss = [i for i, bb in enumerate(bb_list) if bb == 0]
        elif subset == 'all':
            ss = [i for i, bb in enumerate(bb_list) if bb >= 0]
        else:
            ss = [i for i, sz in enumerate(area_list)
                if self.min_sizes[subset] <= sz < self.max_sizes[subset] and bb_list[i] > 0]
        
        if len(ss) == 0:
            return [], [], [], [], []

        name = [name_list[i] for i in ss]
        area = [area_list[i] for i in ss]
        bbox = [bb_list[i] for i in ss]
        ciou = [ciou_list[i] for i in ss]
        conf = [conf_list[i] for i in ss]

        return name, area, bbox, ciou, conf

    def finalize_stats(self):
        name_full_list, area_full_list, bb_full_list, ciou_full_list, confidence_full_list = self.gather_results()
        import pdb;pdb.set_trace()
        metrics = {}
        for iou_thr in self.iou_thrs:
            for subset in ['all', 'visible']:
                _, _, bb_list, ciou_list, conf_list = self.filter_subset(subset, name_full_list, area_full_list, bb_full_list, ciou_full_list, confidence_full_list)
                subset_name = f'{subset}@{int(iou_thr*100)}' if subset is not None else f'@{int(iou_thr*100)}'
                if len(ciou_list) == 0:
                    p, r, ap, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    p, r = self.calc_precision_recall(bb_list, ciou_list, conf_list, -1000, iou_thr)
                    ap = self.calc_ap(bb_list, ciou_list, conf_list, iou_thr)
                    auc = self.cal_auc(bb_list, ciou_list)

                    conf_thr = list(sorted(conf_list))[::max(1,len(conf_list)//10)]
                    pr = [self.calc_precision_recall(bb_list, ciou_list, conf_list, thr, iou_thr) for thr in conf_thr]
                    f1 = [2*r*p/(r+p) if r+p>0 else 0. for p, r in pr]
                metrics[f'Precision-{subset_name}'] = p
                # metrics[f'Recall-{subset_name}'] = r
                if np.isnan(f1).any():
                    metrics[f'F1-{subset_name}'] = f1
                else:
                    metrics[f'F1-{subset_name}'] = ' '.join([f'{f*100:.1f}' for f in f1])
                metrics[f'AP-{subset_name}'] = ap
                metrics[f'AUC-{subset_name}'] = auc

        return metrics

    def gather_results(self):
        import torch.distributed as dist
        if not dist.is_initialized():
            return self.name_list, self.area_list, self.bb_list, self.ciou_list, self.confidence_list
        world_size = dist.get_world_size()

        bb_list = [None for _ in range(world_size)]
        dist.all_gather_object(bb_list, self.bb_list)
        bb_list = [x for bb in bb_list for x in bb]

        area_list = [None for _ in range(world_size)]
        dist.all_gather_object(area_list, self.area_list)
        area_list = [x for area in area_list for x in area]

        ciou_list = [None for _ in range(world_size)]
        dist.all_gather_object(ciou_list, self.ciou_list)
        ciou_list = [x for ciou in ciou_list for x in ciou]

        confidence_list = [None for _ in range(world_size)]
        dist.all_gather_object(confidence_list, self.confidence_list)
        confidence_list = [x for conf in confidence_list for x in conf]

        name_list = [None for _ in range(world_size)]
        dist.all_gather_object(name_list, self.name_list)
        name_list = [x for name in name_list for x in name]

        return name_list, area_list, bb_list, ciou_list, confidence_list

    def precision_at_50(self):
        ss = [i for i, bb in enumerate(self.bb_list) if bb > 0]
        return np.mean(np.array([self.ciou_list[i] for i in ss])>0.5)

    def precision_at_50_object(self):
        max_num_obj = max(self.bb_list)
        for num_obj in range(1, max_num_obj+1):
            ss = [i for i, bb in enumerate(self.bb_list) if bb == num_obj]
            precision = np.mean(np.array([self.ciou_list[i] for i in ss])>0.5)
            print('\n'+f'num_obj:{num_obj}, precision:{precision}')

    def f1_at_50(self):
        # conf_thr = np.array(self.confidence_list).mean()
        p, r = self.calc_precision_recall(self.bb_list, self.ciou_list, self.confidence_list, self.default_conf_thr, 0.5)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.

    def ap_at_50(self):
        return self.calc_ap(self.bb_list, self.ciou_list, self.confidence_list, 0.5)

    def clear(self):
        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []

    def update(self, bb, gt, conf, pred, pred_thr, name):
        if isinstance(conf, torch.Tensor):
            conf = conf.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        # Compute binary prediction map
        infer = np.zeros((224, 224))
        infer[pred >= pred_thr] = 1

        # Compute ciou between prediction and ground truth
        ciou = np.sum(infer*gt) / (np.sum(gt) + np.sum(infer * (gt == 0)))

        # Compute ground truth size
        area = gt.sum()

        # Save
        self.confidence_list.append(conf)
        self.ciou_list.append(ciou)
        self.area_list.append(area)
        self.name_list.append(name)
        self.bb_list.append(len(bb))