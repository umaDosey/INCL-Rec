# INCL: Inductive Contrastive Learning for CF

This is our official implementation for these papers:

Yuma Dose, Shuichiro Haruta, and Takahiro Hara
"A Graph-Based Recommendation Model Using Contrastive Learning for Inductive Scenario"
Proceedings of IEEE International Conference on Machine Learning and Applications (ICMLA), pages 174-181, 2023.

Yuma Dose, Shuichiro Haruta, and Takahiro Hara
"INCL: A Graph-based Recommendation Model Using Contrastive Learning for Inductive Scenario"
情報処理学会論文誌, volume 65, number 11, pages 1-9, November 2024.

## Environment

Python 3.8

Pytorch >= 1.8

DGL >= 0.8

## Dataset

The processed data of Gowalla, Yelp, and Amazon-book can be downloaded in [Baidu Wangpan](https://pan.baidu.com/s/18VcjV_HLhf9FcKgr3-tusQ) with code 1189, or [Google Drive](https://drive.google.com/file/d/1BAN5MJXtRinHTypsszgpTMIJx2RaSj54/view?usp=sharing).

Please place the processed data like:

```
├─igcn_cf
│  ├─data
│  │  ├─Gowalla
|  |  | |─time
|  |  | |-0
|  |  | └─...
│  │  |─Yelp
│  │  └─Amazon
│  |─run
|  └─...
```

For each dataset, **time** folder contains the dataset splitted by time, which is used to tune hyperparameters. 

**0,1,2,3,4** are five random splitted datasets, which are used to train in the transductive scenario and evaluate. 

**0_dropit** is the reduced version of **0** with fewer interactions, which is used to train in the inductive scenario with new interactions.  

**0_dropui** is the reduced version of **0** with fewer users and fewer items, which is used to train in the inductive scenario with new users/items. 


## Quick Start

To launch the experiment in different scenario settings, you can use:

```
python -u -m run.run
python -u -m run.dropit.igcn_dropit
python -u -m run.dropui.igcn_dropui
```

