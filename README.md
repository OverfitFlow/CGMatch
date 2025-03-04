# CGMatch

This repository hosts the official implementation of the **CVPR 2025** paper **"CGMatch: A Different Perspective on Semi-supervised Learning"**. CGMatch addresses a critical challenge in semi-supervised learning (SSL) where conventional methods struggle with severely limited labeled data (4, 10 and 25 labels per class). In the paper, a new metric known as Count-Gap (CG) is introduced to effectively discover unlabeled examples beneficial for model training for the fist time.

**Prerequisites**

Our implementation builds upon USB (https://github.com/microsoft/Semi-supervised-learning), with modifications reflected in the uploaded files. To install the required packages, please refer to the description in the USB.

**Usage**

To train using CGMatch, we take the dataset CIFAR10 with 40 labels as an example, please run:
```
python train.py --c config/classic_cv/cgmatch/cgmatch_cifar10_40_0.yaml
```
