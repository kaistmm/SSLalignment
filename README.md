# Sound Source Localization is All About Alignment (ICCV’23)

Official PyTorch implementation of our following papers:

>  **[Sound Source Localization is All About Cross-Modal Alignment](https://openaccess.thecvf.com/content/ICCV2023/papers/Senocak_Sound_Source_Localization_is_All_about_Cross-Modal_Alignment_ICCV_2023_paper.pdf)**  
>
> [Arda Senocak*](https://ardasnck.github.io/), [Hyeonggon Ryu*](https://sites.google.com/view/hyeonggonryu), [Junsik Kim*](https://sites.google.com/site/jskimcv/), [Tae-Hyun Oh](https://ami.postech.ac.kr/members/tae-hyun-oh), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people), [Joon Son Chung](https://mmai.io/joon/) (* Equal Contribution)
>
>  ICCV 2023

>  **[Aligning Sight and Sound: Advanced Sound Source Localization Through Audio-Visual Alignment](https://arxiv.org/abs/2407.13676)**  
>
> [Arda Senocak*](https://ardasnck.github.io/), [Hyeonggon Ryu*](https://sites.google.com/view/hyeonggonryu), [Junsik Kim*](https://sites.google.com/site/jskimcv/), [Tae-Hyun Oh](https://ami.postech.ac.kr/members/tae-hyun-oh), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people), [Joon Son Chung](https://mmai.io/joon/) (* Equal Contribution)
>
>  (Minor Revision) TPAMI 2024

## Index
- [Overview](#overview)
- [Interactive Synthetic Sound Source (IS3) Dataset](#interactive-synthetic-sound-source-is3-dataset)
- [Environment](#environment)
- [Model Checkpoints](#model-checkpoints)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)

## Overview

<div align="center">
    <img src="./figs/teaser.png" alt="Pipeline" style="width: 75%;"/>
</div>

## Interactive Synthetic Sound Source (IS3) Dataset
![is3](./figs/is3.png)

*IS3 dataset is available [`here`](https://drive.google.com/file/d/1j-2sY6aJMS9kTpamaJM-4vg9eI-40VFB/view?usp=sharing)*

The IS3 data is organized as follows:

Note that in IS3 dataset, each annotation is saved as a separate file. For example; the sample `accordion_baby_10467` image contains two annotations for accordion and baby objects. These annotations are saved as `accordion_baby_10467_accordion` and `accordion_baby_10467_baby` for straightforward use. You can always project bounding boxes or segmentation maps onto the original image to see them all at once.

`images` and `audio_waw` folders contain all the image and audio files respectively. 

`IS3_annotation.json` file contains ground truth bounding box and category information of each annotation.

`gt_segmentation` folder contains segmentation maps in binary image format for each annotation. You can query the file name in `IS3_annotation.json` to get semantic category of each segmentation map.

## Environment

## Model Checkpoints
The model checkpoints are available for the following experiments:

| Training Set                  | Test Set | Model Type | Performance (cIoU) | Checkpoint |
|--------------------------|:-------------:|:-------------:|:-------------:|:------------:|
| VGGSound-144K           |  VGG-SS       | NN w/ Sup. Pre. Enc.        | 39.94      | [Link](https://drive.google.com/file/d/1-WUEoAmp4WBj4Tbsi9Ybh2bq4W6-uYnd/view?usp=drive_link) |
| VGGSound-144K           |  VGG-SS       | NN w/ Self-Sup. Pre. Enc.        | 39.16       | [Link](https://drive.google.com/file/d/1p_eXOlZfeCo5EwY5RRkh4YnAnMZJ0Eyz/view?usp=drive_link) |
| VGGSound-144K           |  VGG-SS       | NN w/ Sup. Pre. Enc. Pre-trained Vision       | 41.42       | [Link](https://drive.google.com/file/d/1FYv6Pt8k8MdHlBHDCaMVGtWZN_UXh_GJ/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Sup. Pre. Enc.        | 85.20      | [Link](https://drive.google.com/file/d/1R_LEEcUEnwREvt_ducCdnEVUI6VHvnqv/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Self-Sup. Pre. Enc.        | 84.80      | [Link](https://drive.google.com/file/d/1HHnUc3sERGrCjUbHklS9uDdA5yxjRtKB/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Sup. Pre. Enc. Pre-trained Vision         | 86.00      | [Link](https://drive.google.com/file/d/1zs0gr1_QVfonw0Q2VVbExwUUDhJSUH35/view?usp=drive_link) |

## Inference

Put checkpoint files into the 'checkpoints' directory:
```
inference
│
└───checkpoints
│       ours_sup_previs.pth.tar
│       ours_sup.pth.tar
│       ours_selfsup.pth.tar
│   test.py
│   datasets.py
│   model.py
```

To evaluate a trained model run

```
python test.py --testset {testset_name} --pth_name {pth_name}
```

| Test Set                 |     testset_name         |
|--------------------------|--------------------------|
| VGG-SS                   |           vggss          |
| Flickr-SoundNet          |           flickr         |
| IS3                      |           is3         |

## Evaluate other methods
Simply save the checkpoint files from the methods as '{method_name}_{put_your_own_message}.pth', such as 'ezvsl_flickr.pth'. We have already handled the trivial settings.
| Paper title                  |  pth_name must contains  |
|--------------------------|--------------------------|
| Localizing Visual Sounds the Hard Way (CVPR 21) [[Paper]](https://arxiv.org/pdf/2104.02691) |           lvs            |
| Localizing Visual Sounds the Easy Way (ECCV 22) [[Paper]](https://arxiv.org/pdf/2203.09324)   |          ezvsl           |
| A Closer Look at Weakly-Supervised Audio-Visual Source Localization (NeurIPS 22) [[Paper]](https://arxiv.org/pdf/2209.09634)   |           slavc          |
| Exploiting Transformation Invariance and Equivariance for Self-supervised Sound Localisation (ACMMM 22) [[Paper]](https://arxiv.org/pdf/2206.12772v2)  |          ssltie         |
| Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning (CVPR 23) [[Paper]](https://arxiv.org/pdf/2303.11302)   |           fnac         |

Example
```
python test.py --testset flickr --pth_name ezvsl_flickr.pth
```


## Training
Training code is coming soon!

## Citation
If you find this code useful, please consider giving a star ⭐ and citing us:

```bibtex
@inproceedings{senocak2023sound,
  title={Sound source localization is all about cross-modal alignment},
  author={Senocak, Arda and Ryu, Hyeonggon and Kim, Junsik and Oh, Tae-Hyun and Pfister, Hanspeter and Chung, Joon Son},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7777--7787},
  year={2023}
}
```
If you use this dataset, please consider giving a star ⭐ and citing us:

```bibtex
@article{senocak2024align,
  title={Aligning Sight and Sound: Advanced Sound Source Localization Through Audio-Visual Alignment},
  author={Senocak, Arda and Ryu, Hyeonggon and Kim, Junsik and Oh, Tae-Hyun and Pfister, Hanspeter and Chung, Joon Son},
  journal={arXiv preprint arXiv:2407.13676},
  year={2024}
}
```
