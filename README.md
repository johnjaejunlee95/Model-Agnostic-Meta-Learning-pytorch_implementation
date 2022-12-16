# MAML-learn2learn
Unofficial pytorch MAML implementation (with learn2learn package: http://learn2learn.net/)


## Dependency
```
pip install numpy
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```  

## Installation
First, **download the learn2learn** package:

```
pip install learn2learn
```

Next, **download the datasets** (reference from https://github.com/renmengye/few-shot-ssl-public):

`mini-ImageNet`: [[Google Drive Link]](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view "because google drive policy has changed, must download manually")\
`tiered-ImageNet`: [[Google Drive Link]](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view "because google drive policy has changed, must download manually")


## Training
```
python miniimagenet_train.py # to run MAML with miniimagenet datasets
python tieredimagenet_train.py # to run MAML with tieredimagenet datasets
```
```
option arguments:  
--epoch:            epoch number (default: 60000)  
--n_way:            N-way (default: 5)  
--k_spt:            K-shot (default: 1)  
--k_qry:            number of query samples (default: 15)  
--imgsz:            resizing images (--> 3*84*84) (default: 84)  
--imgc:             RGB(image channel) (default: 3)  
--num_filters :     size of convolution filters (default: 32)  
--task_num:         meta-batch size (default: 4)  
--val_task:         number of tasks for evaluation (default: 600)  
--meta_lr:          outer-loop learning rate (default: 1e-3)  
--update_lr:        inner-loop learning rate (default: 1e-2)  
--update_step:      number of inner-loop update steps while training (default: 5)  
--update_test_step: number of inner-loop update steps while evaluating (default: 10)  
--version:          saving file version (default: 0)  
--datasets_root:    root of datatsets  (default: '/data01/jjlee_hdd/data')  
```

## Result
**`mini-ImageNet`**
|  | 5-Way 1-Shot | 5-Way 5-Shot |  
|------|---|---|
| Original | 48.7 $\pm$ 1.84%| 63.11 $\pm$ 0.92% |  
| Ours (reproduced) | 47.93 $\pm$ 1.85% | 62.43 $\pm$ 0.89% |  
  
**`tiered-ImageNet`** (Use wider 4-ConvBlock: filter_size = 64)
|  | 5-Way 1-Shot | 5-Way 5-Shot |  
|------|---|---|  
| Original | 51.67 $\pm$ 1.81% | 70.30 $\pm$ 1.75% |  
| Ours (reproduced) | tba | tba |  
