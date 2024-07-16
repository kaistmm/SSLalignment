# Sound Source Localization is All About Alignment (ICCV’23)

Official PyTorch implementation of our following papers:

>  **[Sound Source Localization is All About Cross-Modal Alignment](https://openaccess.thecvf.com/content/ICCV2023/papers/Senocak_Sound_Source_Localization_is_All_about_Cross-Modal_Alignment_ICCV_2023_paper.pdf)**  
>
> [Arda Senocak*](https://ardasnck.github.io/), [Hyeonggon Ryu*](https://sites.google.com/view/hyeonggonryu), [Junsik Kim*](https://sites.google.com/site/jskimcv/), [Tae-Hyun Oh](https://ami.postech.ac.kr/members/tae-hyun-oh), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people), [Joon Son Chung](https://mmai.io/joon/) (* Equal Contribution)
>
>  ICCV 2023

>  **[Aligning Sight and Sound: Advanced Sound Source Localization Through Audio-Visual Alignment](https://openaccess.thecvf.com/content/ICCV2023/papers/Senocak_Sound_Source_Localization_is_All_about_Cross-Modal_Alignment_ICCV_2023_paper.pdf)**  
>
> [Arda Senocak*](https://ardasnck.github.io/), [Hyeonggon Ryu*](https://sites.google.com/view/hyeonggonryu), [Junsik Kim*](https://sites.google.com/site/jskimcv/), [Tae-Hyun Oh](https://ami.postech.ac.kr/members/tae-hyun-oh), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people), [Joon Son Chung](https://mmai.io/joon/) (* Equal Contribution)
>
>  arXiV 2024

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

*IS3 dataset is available [`here`](https://drive.google.com/file/d/12jq7-Ke09ZPoUI1od44q97DNLrThoHc3/view?usp=sharing)*

## Environment

## Model Checkpoints
The model checkpoints are available for the following experiments:

| Training Set                  | Test Set | Model Type | Performance (cIoU) | Checkpoint |
|--------------------------|:-------------:|:-------------:|:-------------:|:------------:|
| VGGSound-144K           |  VGG-SS       | NN w/ Sup. Pre. Enc.        | 39.94      | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| VGGSound-144K           |  VGG-SS       | NN w/ Self-Sup. Pre. Enc.        | 39.16       | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| VGGSound-144K           |  VGG-SS       | NN w/ Sup. Pre. Enc. Pre-trained Vision       | 41.42       | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Sup. Pre. Enc.        | 85.20      | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Self-Sup. Pre. Enc.        | 84.80      | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| Flickr-SoundNet-144K           |  Flickr-SoundNet       | NN w/ Sup. Pre. Enc. Pre-trained Vision         | 86.00      | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |

## Inference

## Training


## Citation
If you find this code useful, please consider citing us:

```bibtex
@inproceedings{senocak2023sound,
  title={Sound source localization is all about cross-modal alignment},
  author={Senocak, Arda and Ryu, Hyeonggon and Kim, Junsik and Oh, Tae-Hyun and Pfister, Hanspeter and Chung, Joon Son},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7777--7787},
  year={2023}
}
```
If you use this dataset, please consider citing us:

```bibtex
@article{senocak2024align,
  title={Aligning Sight and Sound: Advanced Sound Source Localization Through Audio-Visual Alignment},
  author={Senocak, Arda and Ryu, Hyeonggon and Kim, Junsik and Oh, Tae-Hyun and Pfister, Hanspeter and Chung, Joon Son},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
