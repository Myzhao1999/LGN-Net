# LGN-Net
## LGN-Net: Local-Global Normality Network for Video Anomaly Detection


<div align="center"><img width="98%" src="https://github.com/Myzhao1999/LGN-Net/blob/main/lgn.png" /></div>


> 
This repository contains the official PyTorch implementation of the following paper:
> **LGN-Net: Local-Global Normality Network for Video Anomaly Detection ()**<br>
> Mengyang Zhao, Xinhua Zeng, Yang Liu, Jing Liu, Di Li, Xing Hu, Chengxin Pang<br>
> Paper:[(https://arxiv.org/abs/2211.07454)](https://arxiv.org/abs/2211.07454)<br>
> 
> **Abstract** *Video anomaly detection (VAD) has been intensively studied for years because of its potential applications in intelligent video systems. Existing unsupervised VAD methods tend to learn normality from training sets consisting of only normal videos and regard instances deviating from such normality as anomalies. However, they often consider only local or global normality. Some of them focus on learning local spatiotemporal representations from consecutive frames in video clips to enhance the representation for normal events. But powerful representation allows these methods to represent some anomalies and causes missed detections. In contrast, the other methods are devoted to memorizing global prototypical patterns of whole training videos to weaken the generalization for anomalies, which also restricts them to represent diverse normal patterns and causes false alarms. To this end, we propose a two-branch model, Local-Global Normality Network (LGN-Net), to learn local and global normality simultaneously. Specifically, one branch learns the evolution regularities of appearance and motion from consecutive frames as local normality utilizing a spatiotemporal prediction network, while the other branch memorizes prototype features of the whole videos as global normality by a memory module. LGN-Net achieves a balance of representing normal and abnormal instances by fusing local and global normality. The fused normality enables our model more generalized to various scenes compared to exploiting single normality. Experiments demonstrate the effectiveness and superior performance of our method. The code is available online: \href{https://github.com/Myzhao1999/LGN-Net}{https://github.com/Myzhao1999/LGN-Net}.*

## Preparation

### Requirements
- python 3
- pytorch 1.6+
- opencv-python
- scikit-image
- lpips
- numpy

### Datasets
This repository supports UCSD ped2, CUHK Avenue, and ShanghaiTech datasets. 
- [UCSD ped2](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
- [CUHK Avenue](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
- [ShanghaiTech](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

After obtaining the datasets, preprocess the data as image files (refer to below). 
```shell
# Dataset preparation example:
ped2
├── train
│   ├── frames
│   │   ├── 01
│   │   │     ├── 001.jpg
...
│   │   ├── 02
```

## Training the Model
`train.py` saves the weights in `--checkpoint` .

To train the model, run following command:
```shell
# Training example UCSD ped2
python train.py \
--dataset_type 'ped2 \
--dataset_path 'your_dataset_directory'  \
--exp_dir 'your_log_directory'\

```

Descriptions of training parameters are as follows:
- `--dataset_type`: training dataset (ped2, avenue or shanghai')
- `--dataset_path`: your dataset directory
- `--loss_compact`: weight of the feature compactness  loss
- `--loss_separate`: weight of the feature separateness loss
- `--msize`: number of the memory items 
- Refer to `train.py` for the other training parameters

## Testing the Model
`evaluate.py`

To evaluate the model, run following command:
```shell
# Training example UCSD ped2
python evaluate.py \
--dataset_type 'ped2 \
--dataset_path 'your_dataset_directory'  \
--model_dir` 'your_trained_model_directory' \
--m_items_dir` 'your_recorded_memory_items_directory'\

```



Descriptions of testing parameters are as follows:
- `--dataset_type`: training dataset (ped2, avenue or shanghai')
- `--dataset_path`: your dataset directory
- `--lambda`: weight of the PSNR in normality score
- `--gamma`: threshold for test updating
- `--model_dir`: directory of model
- `--m_items_dir`: directory of recorded memory items
- Refer to `evaluate.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models.
- [Pretrained model for UCSD Ped2](https://drive.google.com/drive/folders/1vdg4i7XMtfNEfaCdBS1poqRzmIDO7i9O?usp=sharing)
- [Pretrained model for CUHK Avenue](https://drive.google.com/drive/folders/1vdg4i7XMtfNEfaCdBS1poqRzmIDO7i9O?usp=sharing)
- [Pretrained model for ShanghaiTech](https://drive.google.com/drive/folders/1vdg4i7XMtfNEfaCdBS1poqRzmIDO7i9O?usp=sharing)

Note that, you should set the correct lambda and gamma values  for different datasets. See more details in the paper.
## Citation
If you find this work useful in your research, please cite the paper:
```

@article{arXiv:2211.07454,
  title={LGN-Net: Local-Global Normality Network for Video Anomaly Detection},
  author={Mengyang Zhao, Yang Liu, Jing Liu, Di Li, Xinhua Zeng},
  journal={arXiv preprint arXiv:2211.07454},
  year={2022}
}

```
## References
Thanks for their excellent work and related code.


@inproceedings{park2020learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}

@ARTICLE{wang2021predrnn,  
author={Wang, Yunbo and Wu, Haixu and Zhang, Jianjin and Gao, Zhifeng and Wang, Jianmin and Yu, Philip and Long, Mingsheng}, 
journal={{IEEE} Trans. Pattern Anal. Mach. Intell.},   
title={PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning},  
year={2022},  
pages={1-1}
}

@inproceedings{lee2021video,
  title={Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning},
  author={Lee, Sangmin and Kim, Hak Gu and Choi, Dae Hwi and Kim, Hyung-Il and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
