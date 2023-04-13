# Visual-Semantic-Transformer
这的项目是复现Visual-Semantic Transformer for Scene Text Recognition这篇论文的工作[VST](https://arxiv.org/abs/2112.00948)

本论文中使用视觉特征去关联它的语义信息，这篇文章中一共包括5个关键的模块ConvNet Module(CNN特征提取), Visual Module(视觉建模), Vsalign Module(Visual Semantic Alignment模块)， Iteraction Module(用于两个模态的信息进行类间和类内的交互), Semantic Module(语义推理模块)。
## Required enveriment
这里我列举一下本项目所使用的packages
```
torch==1.1.0
torchvision==0.3.0
fastai==1.0.60
LMDB
Pillow
opencv-python
tensorboardX
```

## 所使用的数据集
- Training datasets

    1. [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/) (MJ): 
        - Use `tools/create_lmdb_dataset.py` to convert images into LMDB dataset
        - [LMDB dataset BaiduNetdisk(passwd:n23k)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
    2. [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST):
        - Use `tools/crop_by_word_bb.py` to crop images from original [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) dataset, and convert images into LMDB dataset by `tools/create_lmdb_dataset.py`
        - [LMDB dataset BaiduNetdisk(passwd:n23k)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)

- Evaluation datasets, LMDB datasets can be downloaded from [BaiduNetdisk(passwd:1dbv)](https://pan.baidu.com/s/1RUg3Akwp7n8kZYJ55rU5LQ), [GoogleDrive](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing).
    1. ICDAR 2013 (IC13)
    2. ICDAR 2015 (IC15)
    3. IIIT5K Words (IIIT)
    4. Street View Text (SVT)
    5. Street View Text-Perspective (SVTP)
    6. CUTE80 (CUTE)

-  `data` 目录的结构是下面的样子：
    ```
    data
    ├── charset_36.txt
    ├── evaluation
    │   ├── CUTE80
    │   ├── IC13_857
    │   ├── IC15_1811
    │   ├── IIIT5k_3000
    │   ├── SVT
    │   └── SVTP
    |── training
    │   ├── MJ
    │   │   ├── MJ_test
    │   │   ├── MJ_train
    │   │   └── MJ_valid
    │   └── ST
    ```

## 模型训练和测试
- 训练模型：
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_vstnet.yaml
  ```
- 模型验证：
  ```
  CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/train_abinet.yaml --phase test
  ```
  附加参数设置:
  - `--checkpoint /path/to/checkpoint` set the path of evaluation model 
  - `--test_root /path/to/dataset` set the path of evaluation dataset
  - `--model_eval [alignment|vision]` which sub-model to evaluate
