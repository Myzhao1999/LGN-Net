## Learning Spatiotemporal Representation Augmented Normality for Video Anomaly Detection


<div align="center"><img width="98%" src="https://github.com/Myzhao1999/SRAN-Net/blob/main/SRAN-Net.png" /></div>


> 
This repository contains the official PyTorch implementation of the following paper:
> **Learning Spatiotemporal Representation Augmented Normality for Video Anomaly Detection (Submitted in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT))**<br>
> Mengyang Zhao, Yang Liu, Xinhua Zeng<br>
> Paper: https://arxiv.org/abs/<br>
> 
> **Abstract** *Video anomaly detection (VAD) has been intensively studied because of its 
>potential to be used in intelligent video systems. In recent years, many unsupervised VAD 
>methods have been proposed, which usually learn normal patterns and consider the instances
> that deviate from such patterns as anomalies. These methods usually tend to memorize the 
>normal patterns in the global training videos or tend to learn local evolution regularities 
>of input video clips. However, almost no methods are able to focus on both local and global 
>features of training videos, which causes these models to have difficulty balancing their 
>representation capability for normal patterns and abnormal patterns. To address this issue,
> we propose to take into account both local spatiotemporal representations and global normal 
>prototype features of training videos and enhance the learned normality utilizing local 
>spatiotemporal representations. Specifically, we devise a two-branch model, in which one
> branch learns the local appearance and motion evolution regularities in current input
> video clips, and the other branch memorizes global normal prototype features of the
> whole training videos.  Experiments on standard benchmarks demonstrate the effectiveness 
>of our proposed method, which boosts the model's capability for representing complex normal
> patterns while limiting the representation capability for abnormal instances. Our method 
>achieves competitive performance compared with the state-of-the-art
> methods with AUCs of 97.1\%, 89.3\%, and 73.0\% on the UCSD Ped2, CUHK Avenue, and ShanghaiTech, respectively.*

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
- [Pretrained model for UCSD Ped2]()
- [Pretrained model for CUHK Avenue]()
- [Pretrained model for ShanghaiTech]()

Note that, you should set the correct lambda and gamma values  for different datasets. See more details in the paper.
## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{
}
```
## References
@inproceedings{park2020learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}
@inproceedings{lee2021video,
  title={Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning},
  author={Lee, Sangmin and Kim, Hak Gu and Choi, Dae Hwi and Kim, Hyung-Il and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}