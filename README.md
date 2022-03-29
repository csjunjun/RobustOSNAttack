# RobustOSNAttack

An official implementation code for paper "Generating Robust Adversarial Examples against Online Social Networks (OSNs)"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


## Background

<p align='center'>  
  <img src='https://github.com/csjunjun/RobustOSNAttack/blob/main/imgs/background.jpg' width='500'/>
</p>
<p align='center'>  
  <em>The scenario of transmitting AEs over OSNs and the impact on their attack capabilities.</em>
</p>

In this paper, we have designed a new framework for generating robust AEs that simultaneously maintain the attack capabilities before and after the OSN transmissions. Our proposed framework consists of a new differentiable network for simulating the OSN and a novel optimization formulation with constraints specifically addressing the attack capabilities. Extensive experiments conducted over Facebook and WeChat demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints. Furthermore, we build a public dataset containing more than 5000 pairs of AEs processed by Facebook or WeChat, facilitating future research on generating AEs.


## Dependency
- torch 1.10.1
- torchattacks 2.14.0
- python 3.8

## Demo

1. train your SIO model
```bash
python train.py 
```
2. generate robust AEs with the pre-trained SIO model
```bash
python rattack.py 
```
**Note: The pretrained weights and dataset can be downloaded from:
[Google Drive](https://drive.google.com/drive/folders/1M-yrL-DDvNd-KV9vxmxxAwAYTVMa6pde?usp=sharing) or 
[Baidu Yun (Code: djzz)](https://pan.baidu.com/s/10GHrwNv57L2d2bD_5X3W0g)**

## Citation

## Acknowledgments
Our code is based on the [DiffJPEG](https://github.com/mlomnitz/DiffJPEG).



